"""Critic branch shards (docs/MODEL9_SEARCH_TEACHER.md, *Policy tails*): the pure-policy tail turns that follow each
search branch's held horizon, written in the teacher-shard layout ``data.load`` reads, so the critic sees states the
search reaches (native-acting, moved, re-mixed, probed) under the policy that will act from them.

Round 3 provenance (provenance.py): each branch's ``meta.json`` records its source checkpoint hash, code commit,
world (seed, withheld, warmup), fork turn, alternative kind, the searched village's personality and the calibration
flag; ``load`` checks the (checkpoint, commit) mix.

One directory per branch (seed, decision point, alternative). Each row is one lineage settlement's tail decision: its
observation arrays, privileged features and personality, the outcome keys ``data.reward_components`` needs, the
branch's unique world id (``world.npy``; sequences never collide across branches), its alternative kind
(``alt_kind.npy``), the turn each settlement was founded on the full path (``founded_turn.npy``, for the lineage
sharing window) and a bootstrap index into the branch's ``bootstrap/`` block: the observation after the last tail turn
of each living settlement. Label arrays are present with every mask false: the actor never trains on these rows.
"""  # noqa: E501 - docstring/prose line kept verbatim

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import contracts, data
from .bridge import observation_batch
from .model import VectorCritic, to_tensors

# World ids of branch rows live far above any seed, so branch and teacher rows never share a
# sequence key.
BRANCH_WORLD_BASE = 1 << 56
TRAJECTORY_FIELDS = (
    "seed",
    "withheld",
    "warmup",
    "point",
    "alternative",
    "fork_turn",
    "camp",
    "bank_entry",
    "snapshot_hash",
)


def trajectory_id(meta: dict) -> str:
    from .provenance import digest

    return digest(
        {k: meta[k] for k in TRAJECTORY_FIELDS} | {"source_run": meta.get("source_run", "unknown")}
    )


def world_id(seed: int, withheld: int, warmup: int, point: int, alternative: int) -> int:
    """A unique int64 world id per (seed, withheld mask, warm-up, decision point, alternative)."""
    if not (
        0 <= point < 1024 and 0 <= alternative < 64 and 0 <= withheld < 16 and 0 <= seed < (1 << 29)
    ):
        raise ValueError("branch world id out of range")
    from .provenance import digest

    return BRANCH_WORLD_BASE + int(digest([seed, withheld, warmup, point, alternative])[:14], 16)


def _observations(
    villages: list[dict[str, Any]], personalities: list[Any]
) -> dict[str, np.ndarray]:
    batch = observation_batch(villages)
    if villages and villages[0].get("privileged") is not None:
        batch["privileged"] = np.asarray([v["privileged"] for v in villages], dtype=np.float32)
    batch["personality"] = np.asarray(personalities, dtype=np.float32)
    return batch


def _dummy_labels(n: int) -> dict[str, np.ndarray]:
    services, strategic = len(contracts.SERVICES), len(contracts.STRATEGIC)
    return {
        "strategic": np.zeros((n, strategic), np.float32),
        "strategic_mask": np.zeros((n, strategic), bool),
        "service_levels": np.zeros((n, services), np.float32),
        "service_mask": np.zeros((n, services), bool),
        "clothing_route": np.full(n, -1, np.int32),
        "clothing_route_label_mask": np.zeros(n, bool),
        "shelter_route": np.full(n, -1, np.int32),
        "shelter_route_label_mask": np.zeros(n, bool),
        "movement": np.zeros(n, np.int32),
        "movement_label_mask": np.zeros(n, bool),
        "fission_policy": np.full(n, -1, np.int32),
        "repair": np.zeros(n, np.float32),
    }


def write(directory: Path, tail: dict[str, Any], meta: dict[str, Any]) -> int:
    """Write one branch's tail (``search_teacher.branch`` returns it as ``tail``) as a shard; returns its rows."""  # noqa: E501 - docstring/prose line kept verbatim
    rows = tail["rows"]
    if not rows:
        return 0
    if (directory / "meta.json").exists():
        raise ValueError(f"branch trajectory already exists: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    arrays = _observations([r["decision"] for r in rows], [r["personality"] for r in rows])
    arrays.update(_dummy_labels(n))
    out = [r["outcome"] for r in rows]
    arrays.update(
        {
            "camp": np.asarray([v["camp"] for v in out], np.int32),
            "turn": np.asarray([r["turn"] for r in rows], np.int32),
            "society": np.asarray([v["society"] for v in out], np.int32),
            "parent_society": np.asarray([v.get("parentSociety", 0) for v in out], np.int32),
            "prior_population": np.asarray([v["priorPopulation"] for v in out], np.int32),
            "population": np.asarray([v["population"] for v in out], np.int32),
            "qol": np.asarray([v["qol"] for v in out], np.float32),
            "nutrition": np.asarray([v["nutrition"] for v in out], np.float32),
            "years": np.asarray([v["years"] for v in out], np.float32),
            "welfare": np.asarray([v["welfare"] for v in out], np.float32),
            "terminated": np.asarray([bool(v["terminated"]) for v in out], bool),
            "trade_utility": np.asarray(
                [float(v.get("tradeUtility", 0.0)) for v in out], np.float32
            ),
            "world": np.full(n, int(meta["world_id"]), np.int64),
            "alt_kind": np.full(n, int(meta["kind"]), np.int32),
            "founded_turn": np.asarray([r["founded_turn"] for r in rows], np.int32),
            "bootstrap_index": np.asarray([r.get("bootstrap_index", -1) for r in rows], np.int32),
            # 1 on a calibration tail (SearchConfig.calibration_tail turns), checked by branch_bias
            # against realised turns.
            "calibration": np.full(n, int(meta.get("calibration", 0)), np.int8),
        }
    )
    for key, value in arrays.items():
        np.save(directory / f"{key}.npy", value)
    (directory / "decision_diagnostics.json").write_text(
        json.dumps(
            [
                {
                    "turn": r["turn"],
                    "camp": r["outcome"]["camp"],
                    "decisionDiagnostics": r["outcome"].get("decisionDiagnostics"),
                }
                for r in rows
            ]
        )
    )
    boot = tail.get("bootstrap") or []
    if boot:
        (directory / "bootstrap").mkdir(exist_ok=True)
        endpoint = _observations([b["village"] for b in boot], [b["personality"] for b in boot])
        endpoint["endpoint_population"] = np.asarray(
            [b["village"]["population"] for b in boot], np.float64
        )
        endpoint["endpoint_world"] = np.full(len(boot), int(meta["world_id"]), np.int64)
        endpoint["endpoint_turn"] = np.full(len(boot), int(tail.get("endpoint_turn", -1)), np.int64)
        for key, value in endpoint.items():
            np.save(directory / "bootstrap" / f"{key}.npy", value)
    from .provenance import checkpoint_hash, digest

    payload = {
        p.relative_to(directory).as_posix(): checkpoint_hash(p)
        for p in sorted(directory.rglob("*.npy"))
    }
    (directory / "meta.json").write_text(
        json.dumps(
            {
                **meta,
                "data_digest": digest(payload),
                "rows": n,
                "branch": True,
                "contract": contracts.NATIVE_CONTRACT,
            }
        )
    )
    return n


def directories(roots: Iterable[Path]) -> list[Path]:
    """Branch directories under each root (``<search out>/branch_shards``), or a root that is itself one."""  # noqa: E501 - docstring/prose line kept verbatim
    out: list[Path] = []
    for root in roots:
        root = Path(root)
        if (root / "meta.json").exists():
            out.append(root)
        elif root.exists():
            out.extend(sorted(p for p in root.iterdir() if (p / "meta.json").exists()))
    return out


def load(
    roots: Iterable[Path],
    provenance: str = "flag",
    report: dict[str, Any] | None = None,
    *,
    purpose: str = "training",
    declared_sources: set[str] | None = None,
    calibration_manifest: dict | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    """All branch rows under ``roots`` and their bootstrap observations; ``bootstrap_index`` is rebased onto the
    concatenated bootstrap block (-1: none). ``provenance`` (provenance.check: reject, flag or allow) applies to the
    shards' (source checkpoint, code commit); ``report`` receives the provenance report."""  # noqa: E501 - docstring/prose line kept verbatim
    from . import provenance as prov

    worlds, boots, offset = [], [], 0
    sources, counts, records, identities, metadata = [], [], [], set(), []
    for d in directories(roots):
        meta = json.loads((d / "meta.json").read_text())
        payload = {
            p.relative_to(d).as_posix(): prov.checkpoint_hash(p) for p in sorted(d.rglob("*.npy"))
        }
        if "data_digest" in meta and meta["data_digest"] != prov.digest(payload):
            raise prov.ProvenanceError(f"branch data changed after generation: {d}")
        if provenance == "reject" and "data_digest" not in meta:
            raise prov.ProvenanceError("branch shard lacks immutable data digest")
        if "source_record" in meta:
            prov.verify_link(meta["source_record"], meta, provenance)
            declared_manifest = prov.source_exclusion(meta["source_record"])
            if meta.get("calibration_manifest_digest", "") != declared_manifest.get(
                "manifest_digest", ""
            ):
                raise prov.ProvenanceError(
                    "branch calibration manifest differs from its source record"
                )
        metadata.append(meta)
        held_manifest = prov.source_exclusion(meta.get("source_record", {}))
        if purpose == "training":
            from .calibration import assert_training

            assert_training(held_manifest, [meta["seed"]], [meta.get("snapshot_hash")])
            if calibration_manifest is not None:
                assert_training(calibration_manifest, [meta["seed"]], [meta.get("snapshot_hash")])
            if meta.get("heldout_calibration", False):
                raise prov.ProvenanceError(
                    "held-out calibration trajectory cannot train the critic"
                )
        identity = meta.get("trajectory_id") or str(d.resolve())
        if "trajectory_id" in meta and identity != trajectory_id(meta):
            raise prov.ProvenanceError("branch trajectory metadata changed after generation")
        if identity in identities:
            raise prov.ProvenanceError(f"duplicate branch trajectory: {identity}")
        identities.add(identity)
        records.append(meta.get("source_record", {}))
        w = data.normalize_optional(data.load_world(d))
        if len(w["camp"]) == 0:
            continue
        sources.append(prov.key(meta))
        counts.append(len(w["camp"]))
        # A directory is one branch. Allocate fresh sequence keys even when independent runs reused
        # saved IDs.
        w["world"] = np.full(len(w["camp"]), BRANCH_WORLD_BASE + len(worlds), np.int64)
        w.setdefault(
            "calibration", np.zeros(len(w["camp"]), np.int8)
        )  # shards written before calibration tails
        index = w["bootstrap_index"].astype(np.int64)
        w["endpoint_turn"] = np.full(len(w["camp"]), -1, np.int64)
        if (d / "bootstrap" / "core.npy").exists():
            b = {p.stem: np.load(p) for p in sorted((d / "bootstrap").glob("*.npy"))}
            b["endpoint_world"] = np.full(len(b["core"]), BRANCH_WORLD_BASE + len(worlds), np.int64)
            if "endpoint_turn" in b:
                w["endpoint_turn"][:] = int(b["endpoint_turn"][0])
            boots.append(b)
            w["bootstrap_index"] = np.where(index >= 0, index + offset, -1)
            offset += len(b["core"])
        else:
            w["bootstrap_index"] = np.full(len(index), -1, np.int64)
        worlds.append(w)
    if not worlds:
        raise ValueError("no branch shards found")
    validated = prov.validate_records(records, provenance)
    checked = {
        **prov.check(sources, provenance, "branch shards", counts, declared_sources),
        **validated,
        "trajectories": len(identities),
        "records": records,
        "branches": metadata,
        "calibration_manifest_digest": (calibration_manifest or {}).get("manifest_digest"),
    }
    if report is not None:
        report.update(checked)
    keys = [k for k in worlds[0] if all(k in w for w in worlds)]
    rows = {k: np.concatenate([w[k] for w in worlds]) for k in keys}
    boot = None
    if boots:
        bkeys = [k for k in boots[0] if all(k in b for b in boots)]
        boot = {k: np.concatenate([b[k] for b in boots]) for k in bkeys}
    return rows, boot


def _head_values(
    critic, batch: dict[str, np.ndarray], normalizer, batch_size: int = 4096
) -> dict[str, np.ndarray]:
    """The critic's per-resident (or raw, for qol_years) expectation per head; a scalar critic as ``welfare``."""  # noqa: E501 - docstring/prose line kept verbatim
    parts: dict[str, list[np.ndarray]] = {}
    n = len(batch["core"])
    with torch.no_grad():
        for start in range(0, n, batch_size):
            sub = {k: v[start : start + batch_size] for k, v in batch.items()}
            out = critic(to_tensors(sub, normalizer))
            heads = out if isinstance(out, dict) else {"welfare": out}
            for name, logits in heads.items():
                parts.setdefault(name, []).append(
                    critic.expectation(logits).numpy().astype(np.float64)
                )
    return {name: np.concatenate(p) for name, p in parts.items()}


def head_scale(name: str, population: np.ndarray) -> np.ndarray:
    from .ppo import head_scale as scale

    return scale(name, population)


def endpoint_values(critic, boot, normalizer) -> dict:
    """All living endpoint settlements, including daughters founded on the final reward turn."""
    if boot is None or not all(
        k in boot for k in ("endpoint_world", "endpoint_turn", "endpoint_population")
    ):
        return {}
    heads = _head_values(critic, boot, normalizer)
    out = {}
    for world, turn in set(
        zip(boot["endpoint_world"].tolist(), boot["endpoint_turn"].tolist(), strict=False)
    ):
        mask = (boot["endpoint_world"] == world) & (boot["endpoint_turn"] == turn)
        for name, u in heads.items():
            out[world, turn, name] = float(
                (u[mask] * head_scale(name, boot["endpoint_population"][mask])).sum()
            )
    return out


def values(
    critic, rows: dict[str, np.ndarray], boot: dict[str, np.ndarray] | None, normalizer
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Raw-unit values per head of every row, and of each row's bootstrap observation (NaN where it has none)."""  # noqa: E501 - docstring/prose line kept verbatim
    u = _head_values(critic, rows, normalizer)
    value = {name: x * head_scale(name, rows["prior_population"]) for name, x in u.items()}
    bootstrap = {name: np.full(len(rows["camp"]), np.nan) for name in u}
    index = rows["bootstrap_index"].astype(np.int64)
    has = index >= 0
    if boot is not None and has.any():
        ub = _head_values(critic, boot, normalizer)
        for name, x in ub.items():
            # The bootstrap observation is taken at the row's end population.
            bootstrap[name][has] = x[index[has]] * head_scale(name, rows["population"][has])
    return value, bootstrap


def targets(
    critic,
    rows: dict[str, np.ndarray],
    boot: dict[str, np.ndarray] | None,
    normalizer,
    lam: float = 0.95,
) -> dict[str, np.ndarray] | np.ndarray:
    """Critic targets for branch rows: lambda-returns over each branch sequence with the current critic's values and
    its bootstrap after the last tail turn, in the critic's own units (per resident, qol_years raw). A vector critic
    gets one target per head (a dict); a scalar critic its welfare target (an array)."""  # noqa: E501 - docstring/prose line kept verbatim
    value, bootstrap = values(critic, rows, boot, normalizer)
    components = data.reward_components(rows)
    out = {
        name: data.lambda_returns(rows, components[name], value[name], lam, bootstrap[name])
        / head_scale(name, rows["prior_population"])
        for name in value
    }
    return out if isinstance(critic, VectorCritic) else out["welfare"]
