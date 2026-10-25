"""Manifest-driven heuristic labor teacher collection into immutable shards.

One shard per fitting world, each collected in its own process through the
production actor-labor bridge (movement on by default) with the shadow
heuristic economic diagnostics. Every row carries world/camp/turn/attempt
identity, the converted action-12 label with per-head validity, the planned
staffing ledger, the quarantine flag for rejected initial recommendations and
the transition rewards for heuristic critic returns.

Publication follows ``teacher_data.py``: arrays first, ``meta.json`` (payload
hashes) last per shard, and ``dataset.json`` only after every shard published.
An existing output is never overwritten.

Example::

    python -m anthropocene_rl.autonomous_v7.labor_teacher_data \\
        --normalizer-checkpoint normalizer.pt --worlds 5000-5031 --turns 25 \\
        --players 16 --jobs 8 --out datasets/labor-teacher-g1-5000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .labor_personality import LaborPersonalityConditioning, personality_for_world
from .labor_bc import (
    LABOR_LABEL_CONTRACT,
    LaborResetOptions,
    collect_heuristic_world,
    validate_fitting_worlds,
    world_balanced_weights,
)
from .labor_model import LaborPersonality
from .protocol import scan, seed_list
from .provenance import ProvenanceError, checkpoint_hash, code_commit, digest, simulator_identity

SCHEMA = 1
SOURCE_KIND = "native_heuristic_labor_teacher"
ARRAY_NAME = re.compile(r"[a-z][a-z0-9_]*")
_TORCH_DTYPES = {
    "bool": torch.bool,
    "float32": torch.float32,
    "float64": torch.float64,
    "int64": torch.long,
    "int32": torch.int32,
}


def load_normalizer(path: Path) -> Any:
    """A bounded normalizer from a normalizer file or a labor trainer checkpoint."""
    from .labor_model import BOUNDED_FEATURE_CONTRACT
    from .model import load_checkpoint_normalizer

    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, Mapping) and "labor_encoder_layout" in state:
        from .labor_training import _models_from_checkpoint

        return _models_from_checkpoint(state)[-1]
    return load_checkpoint_normalizer(state, expected_feature_contract=BOUNDED_FEATURE_CONTRACT)


def normalizer_digest(normalizer: Any) -> str:
    from .labor_training import _normalization_digest

    return _normalization_digest(normalizer)


def source_record(
    options: Mapping[str, Any],
    schema: Any,
    bridge_sha256: str,
    *,
    production: bool,
    source_kind: str = SOURCE_KIND,
) -> dict[str, Any]:
    scan(dict(options))
    learner = code_commit()
    if production and not re.fullmatch(r"[0-9a-f]{40}", learner):
        raise ProvenanceError("labor teacher generation requires a clean learner commit")
    body = {
        "schema": SCHEMA,
        "source_kind": source_kind,
        "label_contract": LABOR_LABEL_CONTRACT,
        "learner_commit": learner,
        "simulator": simulator_identity(schema, production=production),
        "bridge_sha256": bridge_sha256,
        "options": dict(options),
        "production": bool(production),
        "flags": {
            "sim_rules": os.environ.get("ANTHROPOCENE_SIM_RULES", ""),
            "sim_statics": os.environ.get("ANTHROPOCENE_SIM_STATICS", ""),
        },
    }
    return dict(body, source_run=digest(body))


def _payload(directory: Path) -> dict[str, Any]:
    out = {}
    for path in sorted(directory.glob("*.npy")):
        array = np.load(path, allow_pickle=False)
        out[path.name] = {
            "sha256": checkpoint_hash(path),
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }
    return out


def write_shard(directory: Path, arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any]) -> dict:
    """Arrays first, meta.json last; an existing directory is never reused."""
    scan(dict(meta))
    if directory.exists():
        raise ProvenanceError("labor teacher shard already exists; no overwrite")
    rows = {len(array) for array in arrays.values()}
    if len(rows) != 1 or next(iter(rows)) <= 0:
        raise ProvenanceError("labor teacher shard arrays must share a positive row count")
    directory.mkdir(parents=True)
    for key, array in sorted(arrays.items()):
        if not ARRAY_NAME.fullmatch(key) or array.dtype.hasobject:
            raise ProvenanceError(f"invalid labor teacher array {key!r}")
        np.save(directory / (key + ".npy"), array, allow_pickle=False)
    record = dict(meta, schema=SCHEMA, rows=int(next(iter(rows))))
    record["payload"] = _payload(directory)
    record["data_digest"] = digest(record["payload"])
    (directory / "meta.json.tmp").write_text(json.dumps(record, sort_keys=True, allow_nan=False))
    os.replace(directory / "meta.json.tmp", directory / "meta.json")
    return record


def verify_shard(directory: Path) -> dict:
    if not (directory / "meta.json").is_file():
        raise ProvenanceError("labor teacher shard not published")
    meta = json.loads((directory / "meta.json").read_text())
    actual = _payload(directory)
    if (
        meta.get("schema") != SCHEMA
        or not actual
        or actual != meta.get("payload")
        or digest(actual) != meta.get("data_digest")
    ):
        raise ProvenanceError("labor teacher shard payload hash/dtype/shape mismatch")
    if any(entry["shape"][0] != meta["rows"] for entry in actual.values()):
        raise ProvenanceError("labor teacher shard row counts differ")
    worlds = np.load(directory / "row_world.npy")
    if set(worlds.tolist()) != {meta["world_seed"]}:
        raise ProvenanceError("labor teacher shard rows belong to another world")
    columns = [
        np.load(directory / "row_turn.npy").tolist(),
        np.load(directory / "row_camp.npy").tolist(),
        np.load(directory / "row_attempt.npy").tolist(),
    ]
    if (directory / "row_start_index.npy").is_file():
        # Learner-state (DAgger) shards: independent branches of one world may
        # revisit a (turn, camp) identity; the branch start index separates them.
        columns.append(np.load(directory / "row_start_index.npy").tolist())
    identities = list(zip(*columns, strict=True))
    if len(set(identities)) != len(identities):
        raise ProvenanceError("duplicate labor teacher decision identities")
    return meta


def _collect_shard(job: Mapping[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    normalizer = load_normalizer(Path(job["normalizer_checkpoint"]))
    options = LaborResetOptions.from_mapping(job["reset_options"])
    personality: Any = LaborPersonality.from_mapping(job["personality"])
    conditioning = LaborPersonalityConditioning.from_state(job.get("personality_conditioning"))
    if conditioning.alpha is not None:
        # A declared distribution: one keyed draw per founding lineage in this world
        # (docs/LABOR_PERSONALITY.md); heuristic labels stay personality-agnostic.
        personality = personality_for_world(conditioning.assignment(0), int(job["world_seed"]))
    started = time.perf_counter()

    def progress(state: Any) -> None:
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "world": int(job["world_seed"]),
                    "decisions": state.decisions,
                    "rows": len(state.rows),
                    "native_resolves": state.native_resolves,
                    "wall_seconds": round(elapsed, 2),
                }
            ),
            file=sys.stderr,
            flush=True,
        )

    collection = collect_heuristic_world(
        progress=progress,
        normalizer=normalizer,
        world_seed=int(job["world_seed"]),
        turns=int(job["turns"]),
        reset_options=options,
        personality=personality,
    )
    seconds = time.perf_counter() - started
    if not collection.rows:
        raise ValueError(f"world {job['world_seed']} produced no policy rows")
    from .labor_bc import _concat

    tensors = _concat(collection.rows, f"labor teacher world {job['world_seed']}")
    keep = ~tensors["row_quarantined"].reshape(-1)
    tensors["row_weight"] = world_balanced_weights(tensors["row_world"], keep)
    arrays = {key: value.detach().cpu().numpy() for key, value in tensors.items()}
    meta = {
        "source_run": job["source_run"],
        "world_seed": int(job["world_seed"]),
        "turns": int(job["turns"]),
        "reset_options": options.state_dict(),
        "reset_identity": collection.reset_identity,
        "normalizer_digest": job["normalizer_digest"],
        "personality": dict(job["personality"]),
        "decisions": int(collection.decisions),
        "native_resolves": int(collection.native_resolves),
        "wall_seconds": float(seconds),
        "counters": dict(collection.counters),
        "label_unavailable_reasons": dict(sorted(collection.reasons.items())),
        "quarantine": collection.quarantine,
        "surviving_camps": int(collection.surviving_camps),
        "return_tail": "truncated-zero-at-collection-horizon",
    }
    directory = Path(job["out"]) / f"world-{int(job['world_seed'])}"
    record = write_shard(directory, arrays, meta)
    return {
        "path": directory.name,
        "world_seed": int(job["world_seed"]),
        "meta_sha256": checkpoint_hash(directory / "meta.json"),
        "data_digest": record["data_digest"],
        "rows": record["rows"],
        "decisions": record["decisions"],
        "native_resolves": record["native_resolves"],
        "wall_seconds": record["wall_seconds"],
    }


def publish_dataset(directory: Path, source: Mapping[str, Any], shards: Sequence[Mapping[str, Any]]) -> dict:
    body = {
        "schema": SCHEMA,
        "source_record": dict(source),
        "source_run": source["source_run"],
        "shards": {str(s["world_seed"]): dict(s) for s in sorted(shards, key=lambda s: s["world_seed"])},
    }
    if len(body["shards"]) != len(shards):
        raise ProvenanceError("duplicate labor teacher worlds")
    body["dataset_digest"] = digest(body)
    (directory / "dataset.json.tmp").write_text(json.dumps(body, sort_keys=True, allow_nan=False))
    os.replace(directory / "dataset.json.tmp", directory / "dataset.json")
    return body


def verify_dataset(directory: Path) -> dict:
    manifest = json.loads((Path(directory) / "dataset.json").read_text())
    if manifest.get("schema") != SCHEMA or not manifest.get("shards"):
        raise ProvenanceError("labor teacher dataset schema/positive shard coverage missing")
    if manifest.get("dataset_digest") != digest({k: v for k, v in manifest.items() if k != "dataset_digest"}):
        raise ProvenanceError("labor teacher dataset digest mismatch")
    source = manifest["source_record"]
    if source.get("source_run") != digest({k: v for k, v in source.items() if k != "source_run"}):
        raise ProvenanceError("labor teacher source record digest mismatch")
    listed = {entry["path"] for entry in manifest["shards"].values()}
    if listed != {p.name for p in Path(directory).iterdir() if p.is_dir()}:
        raise ProvenanceError("labor teacher dataset contains missing/undeclared shards")
    for world, entry in manifest["shards"].items():
        shard = Path(directory) / entry["path"]
        if checkpoint_hash(shard / "meta.json") != entry["meta_sha256"]:
            raise ProvenanceError("labor teacher shard meta hash mismatch")
        meta = verify_shard(shard)
        if (
            str(meta["world_seed"]) != world
            or meta["source_run"] != manifest["source_run"]
            or meta["data_digest"] != entry["data_digest"]
        ):
            raise ProvenanceError("labor teacher shard source/payload binding mismatch")
    return manifest


def load_labor_dataset(directory: Path, *, expected_normalizer_digest: str | None = None) -> dict[str, Any]:
    """Verify and concatenate shards; recompute world-balanced weights globally."""
    manifest = verify_dataset(Path(directory))
    tensors: dict[str, list[torch.Tensor]] = defaultdict(list)
    counters: dict[str, int] = defaultdict(int)
    reasons: dict[str, int] = defaultdict(int)
    quarantine: list[Any] = []
    digests = set()
    for entry in manifest["shards"].values():
        shard = Path(directory) / entry["path"]
        meta = json.loads((shard / "meta.json").read_text())
        digests.add(meta["normalizer_digest"])
        for name in meta["payload"]:
            array = np.load(shard / name, allow_pickle=False)
            tensors[name[: -len(".npy")]].append(torch.from_numpy(array))
        for key, value in meta["counters"].items():
            counters[key] += int(value)
        for key, value in meta["label_unavailable_reasons"].items():
            reasons[key] += int(value)
        quarantine.extend(meta["quarantine"])
    if len(digests) != 1:
        raise ProvenanceError("labor teacher shards use different normalizers")
    digest_value = next(iter(digests))
    if expected_normalizer_digest is not None and digest_value != expected_normalizer_digest:
        raise ProvenanceError("labor teacher dataset normalizer differs from the checkpoint normalizer")
    dataset: dict[str, Any] = {key: torch.cat(values, dim=0) for key, values in tensors.items()}
    keep = ~dataset["row_quarantined"].reshape(-1)
    dataset["row_weight"] = world_balanced_weights(dataset["row_world"], keep)
    options = manifest["source_record"]["options"]
    dataset["collection"] = {
        "label_contract": LABOR_LABEL_CONTRACT,
        "source_kind": manifest["source_record"].get("source_kind", SOURCE_KIND),
        "actor_only": bool(options.get("actor_only", False)),
        "rows": int(len(keep)),
        "dataset_digest": manifest["dataset_digest"],
        "source_run": manifest["source_run"],
        "normalizer_digest": digest_value,
        "total_decisions": int(counters["total_decisions"]),
        "initial_accepted": int(counters["initial_accepted"]),
        "initial_rejected": int(counters["initial_rejected"]),
        "quarantined_rows": int(counters["quarantined"]),
        "labor_label_valid_rows": int(counters["labor_label_valid"]),
        "full_action_label_valid_rows": int(counters["full_action_label_valid"]),
        "label_unavailable_reasons": dict(sorted(reasons.items())),
        "quarantine_count": len(quarantine),
        "world_seeds": sorted(int(world) for world in manifest["shards"]),
        "turns": int(options["turns"]),
        "reset_options": options["reset_options"],
        "sampling": "world-balanced-row-weights",
    }
    return dataset


# Identity fields that only learner-state shards carry; teacher rows get these
# defaults when several sources are aggregated (row_source 0 = teacher).
ROW_SOURCES = {SOURCE_KIND: 0, "learner_state_dagger": 1}
AGGREGATE_IDENTITY_DEFAULTS = {
    "row_source": 0,
    "row_round": 0,
    "row_depth": 0,
    "row_start": -1,
    "row_start_index": -1,
    "row_branch_step": -1,
}


def source_world_balanced_weights(
    sources: torch.Tensor, worlds: torch.Tensor, keep: torch.Tensor | None = None
) -> torch.Tensor:
    """Equal total mass per source, equal mass per world within a source (mean weight one).

    With one source this equals :func:`labor_bc.world_balanced_weights`.
    """
    sources = sources.reshape(-1)
    worlds = worlds.reshape(-1)
    keep = torch.ones_like(worlds, dtype=torch.bool) if keep is None else keep.reshape(-1).bool()
    weights = torch.zeros(len(worlds), dtype=torch.float32)
    total = int(keep.sum())
    kept_sources = torch.unique(sources[keep]).tolist()
    for source in kept_sources:
        in_source = (sources == source) & keep
        source_worlds = torch.unique(worlds[in_source]).tolist()
        for world in source_worlds:
            rows = in_source & (worlds == world)
            weights[rows] = total / (
                len(kept_sources) * len(source_worlds) * float(rows.sum())
            )
    return weights


def load_labor_datasets(
    directories: Sequence[Path], *, expected_normalizer_digest: str | None = None
) -> dict[str, Any]:
    """One dataset unchanged, or several sources mixed with equal source mass (brief 6.1).

    A single directory returns :func:`load_labor_dataset` exactly. Several
    directories are verified independently, grouped by source kind (the
    original teacher demonstrations versus every aggregated learner-state
    round), concatenated in the given order and reweighted so each source kind
    carries equal total mass and each world carries equal mass within its
    source. Quarantined rows keep zero weight.
    """
    directories = [Path(d) for d in directories]
    if not directories:
        raise ValueError("at least one labor dataset is required")
    if len(directories) == 1:
        return load_labor_dataset(
            directories[0], expected_normalizer_digest=expected_normalizer_digest
        )
    parts = [
        load_labor_dataset(d, expected_normalizer_digest=expected_normalizer_digest)
        for d in directories
    ]
    digests = [p["collection"]["dataset_digest"] for p in parts]
    if len(set(digests)) != len(digests):
        raise ProvenanceError("the same labor dataset is listed more than once")
    if len({p["collection"]["normalizer_digest"] for p in parts}) != 1:
        raise ProvenanceError("aggregated labor datasets use different normalizers")
    tensors_by_part = []
    for part in parts:
        kind = part["collection"]["source_kind"]
        if kind not in ROW_SOURCES:
            raise ProvenanceError(f"unknown labor dataset source kind {kind!r}")
        rows = int(part["row_world"].shape[0])
        tensors = {k: v for k, v in part.items() if isinstance(v, torch.Tensor)}
        for key, default in AGGREGATE_IDENTITY_DEFAULTS.items():
            if key not in tensors:
                value = ROW_SOURCES[kind] if key == "row_source" else default
                tensors[key] = torch.full((rows,), value, dtype=torch.long)
        if not bool((tensors["row_source"].reshape(-1) == ROW_SOURCES[kind]).all()):
            raise ProvenanceError("labor dataset row_source differs from its source kind")
        tensors_by_part.append(tensors)
    keys = set(tensors_by_part[0])
    for tensors in tensors_by_part[1:]:
        if set(tensors) != keys:
            raise ProvenanceError(
                f"aggregated labor datasets have different arrays: {sorted(keys ^ set(tensors))}"
            )
    dataset: dict[str, Any] = {
        key: torch.cat([t[key] for t in tensors_by_part], dim=0) for key in sorted(keys)
    }
    keep = ~dataset["row_quarantined"].reshape(-1)
    dataset["row_weight"] = source_world_balanced_weights(
        dataset["row_source"], dataset["row_world"], keep
    )
    sources: dict[str, Any] = {}
    row_source = dataset["row_source"].reshape(-1)
    for kind, code in sorted(ROW_SOURCES.items(), key=lambda item: item[1]):
        in_source = row_source == code
        if not bool(in_source.any()):
            continue
        kept = in_source & keep
        identity = set(
            zip(
                dataset["row_world"].reshape(-1)[kept].tolist(),
                dataset["row_turn"].reshape(-1)[kept].tolist(),
                dataset["row_camp"].reshape(-1)[kept].tolist(),
                strict=True,
            )
        )
        sources[kind] = {
            "rows": int(in_source.sum()),
            "kept_rows": int(kept.sum()),
            "worlds": sorted(set(dataset["row_world"].reshape(-1)[kept].tolist())),
            "unique_world_turn_camp_support": len(identity),
            "repeated_world_turn_camp_rows": int(kept.sum()) - len(identity),
            "weight_mass": float(dataset["row_weight"][in_source].sum()),
        }
    first = parts[0]["collection"]
    dataset["collection"] = {
        "label_contract": LABOR_LABEL_CONTRACT,
        "source_kind": "aggregated",
        "actor_only": any(p["collection"]["actor_only"] for p in parts),
        "dataset_digest": digest(digests),
        "component_dataset_digests": digests,
        "component_collections": [
            {k: v for k, v in p["collection"].items() if k != "label_unavailable_reasons"}
            for p in parts
        ],
        "normalizer_digest": first["normalizer_digest"],
        "sources": sources,
        "rows": int(len(keep)),
        "quarantined_rows": int((~keep).sum()),
        "world_seeds": sorted(set(dataset["row_world"].reshape(-1).tolist())),
        "sampling": "equal-source-mass-then-world-balanced-row-weights",
    }
    return dataset


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--normalizer-checkpoint", type=Path, required=True)
    parser.add_argument("--worlds", required=True, help="explicit fitting seeds/ranges, e.g. 5000-5031")
    parser.add_argument("--turns", type=int, default=25, help="decisions per world")
    parser.add_argument("--players", type=int, default=8)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--separation", type=int, default=None)
    parser.add_argument("--no-movement", action="store_true", help="named mechanical fixture only")
    parser.add_argument("--qol-weight", type=float, default=0.5)
    parser.add_argument("--population-weight", type=float, default=0.5)
    parser.add_argument("--personality", default="balanced", help="'balanced' or 'dirichlet:ALPHA'")
    parser.add_argument("--personality-key", type=int, default=0)
    parser.add_argument("--personality-drift", type=float, default=200.0)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="development only: permit a dirty learner/simulator identity (recorded as non-production)",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    worlds = validate_fitting_worlds(seed_list(args.worlds), "labor teacher fitting worlds")
    if args.turns < 1 or args.jobs < 1:
        raise ValueError("turns and jobs must be positive")
    if args.out.exists():
        raise ProvenanceError("labor teacher output exists; regenerate into a new directory")
    reset = LaborResetOptions(
        players=args.players,
        movement=not args.no_movement,
        width=args.width,
        height=args.height,
        separation=args.separation,
    )
    personality = LaborPersonality(
        qol_years=args.qol_weight, population_years=args.population_weight
    ).state_dict()
    conditioning = LaborPersonalityConditioning(
        args.personality, args.personality_key, args.personality_drift
    )
    normalizer = load_normalizer(args.normalizer_checkpoint)
    from .bridge import DEFAULT_BRIDGE
    from .labor_bridge import LaborModel9Bridge

    bridge_path = Path(os.environ.get("ANTHROPOCENE_MODEL9_BRIDGE", str(DEFAULT_BRIDGE)))
    bridge_sha = checkpoint_hash(bridge_path)
    if not re.fullmatch(r"[0-9a-f]{64}", bridge_sha):
        raise ProvenanceError("production bridge binary missing")
    options = {
        "worlds": worlds,
        "turns": int(args.turns),
        "reset_options": reset.state_dict(),
        "personality": personality,
        "personality_conditioning": conditioning.state_dict(),
        "normalizer_digest": normalizer_digest(normalizer),
        "normalizer_checkpoint_sha256": checkpoint_hash(args.normalizer_checkpoint),
        "opening_limit": 40,
        "partition": "fitting",
    }
    with LaborModel9Bridge() as probe:
        source = source_record(options, probe.schema, bridge_sha, production=not args.allow_dirty)
    args.out.mkdir(parents=True)
    jobs = [
        {
            "normalizer_checkpoint": str(args.normalizer_checkpoint),
            "normalizer_digest": options["normalizer_digest"],
            "reset_options": reset.state_dict(),
            "personality": personality,
            "personality_conditioning": conditioning.state_dict(),
            "world_seed": world,
            "turns": int(args.turns),
            "source_run": source["source_run"],
            "out": str(args.out),
        }
        for world in worlds
    ]
    started = time.perf_counter()
    if args.jobs == 1:
        shards = [_collect_shard(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as pool:
            shards = list(pool.map(_collect_shard, jobs))
    if checkpoint_hash(bridge_path) != bridge_sha or code_commit() != source["learner_commit"]:
        raise ProvenanceError("labor teacher producer changed during collection; dataset not published")
    manifest = publish_dataset(args.out, source, shards)
    elapsed = time.perf_counter() - started
    rows = sum(s["rows"] for s in shards)
    resolves = sum(s["native_resolves"] for s in shards)
    print(
        json.dumps(
            {
                "dataset": str(args.out),
                "dataset_digest": manifest["dataset_digest"],
                "worlds": len(shards),
                "rows": rows,
                "native_resolves": resolves,
                "wall_seconds": elapsed,
                "rows_per_minute": rows / max(elapsed, 1e-9) * 60.0,
                "native_resolves_per_minute": resolves / max(elapsed, 1e-9) * 60.0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
