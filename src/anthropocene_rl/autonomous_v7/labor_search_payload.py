"""Versioned candidate-payload export of the labor search (intervention B training data).

The advantage head ``A_j(C, a)`` (``labor_advantage``, model side) is trained on
within-root return differences between complete candidate actions restored
from one identical root. The label rows of ``labor_search`` store only the
chosen label per personality; this module defines the separate, versioned
export of *every* evaluated candidate that such training needs.

Schema ``labor-search-candidate-payload-v1`` (``PAYLOAD_SCHEMA_VERSION = 1``)
---------------------------------------------------------------------------

Each search shard ``world-<seed>/`` gains two immutable sub-tables written with
``labor_teacher_data.write_shard`` (arrays first, ``meta.json`` with payload
hashes last) and bound into the shard meta under ``candidate_payload``
(schema, digests, row counts), so the label shard's own hash covers them:

``world-<seed>/advantage-roots/`` -- one row per searched root (focal camp at a
decision). Columns:

* every actor-visible input tensor of the focal camp at the root, exactly as
  the label rows store them (``labor_policy_tensors``), with
  ``labor_personality`` set to the root personality;
* the raw critic-only privileged arrays of the root (``privileged_*``, schema
  in the shard meta) when the producer emits them -- the same decision
  boundary as the observation;
* identity ``row_world``, ``row_camp`` (focal), ``row_turn``, ``row_attempt``,
  ``row_quarantined`` and ``root_index`` (shard-local, the join key);
* ``root_source`` (0 = H12-visited, 1 = actor-visited), ``root_society``,
  ``root_population``, ``root_turn_years``, ``root_personality`` [2],
  ``root_candidates``, ``root_final_horizon`` (decisions reached by the
  re-evaluated survivors), ``root_truncated`` (resolve cap stopped halving),
  ``root_horizons`` [H] (the declared halving horizons), ``root_reeval_streams``
  and ``root_ranker_mode`` (0 = no ranker, every candidate natively evaluated;
  1 = ranked and pruned to top-k plus H12; 2 = ranked but fully evaluated, the
  regret-measurement roots).

``world-<seed>/advantage-candidates/`` -- one row per accepted candidate:

* join/identity: ``cand_root`` (= ``root_index``), ``cand_index`` (0 is H12),
  ``row_world``, ``row_turn``, ``row_camp``, ``cand_family`` (search
  ``FAMILY_CODES``), ``cand_is_h12``, ``cand_chosen_heads`` [6] in
  :data:`HEADS` order, ``cand_digest`` [32] (sha256 of the canonical payload,
  uint8), ``cand_attempt_valid`` (native validation at the root);
* the complete executed action in raw output units, exactly as submitted:
  ``action_specialty_allocation`` [14] (simplex over ``SPECIALTIES``, idle
  first), ``action_activity_allocation`` [13, 22] (worker-time simplex per
  role over ``ACTIVITIES``), ``action_movement_code`` (bridge code) and
  ``action_movement_slot`` (index into :data:`MOVEMENT_UNION_CODES`, the
  actor's 17-way movement union), ``action_fission`` (0..2),
  ``action_trade_gate`` [7], ``action_trade_value`` [7] (relative value,
  already clamped to ``TRADE_VALUE_RANGE``), ``action_trade_share``;
* the same action compiled to native units: ``action_specialty_counts`` [14]
  (largest-remainder residents, sums to ``action_population``) and
  ``action_worker_years`` [13, 22] (residents x worker-time share x
  ``action_turn_years``);
* selection stream 0 (deterministic H12 continuation, shared by every
  candidate): ``cand_natively_evaluated`` (False only for ranker-pruned
  candidates), ``cand_horizon_reached`` (decisions actually executed),
  ``cand_terminated`` (the focal camp left the game inside the branch),
  ``cand_last_round``, ``cand_step_rewards`` [Hmax, 4], ``cand_step_years``
  [Hmax], ``cand_step_mask`` [Hmax], ``cand_returns_by_horizon`` [H, 4] (the
  focal camp's discounted return at each declared horizon, tau =
  ``discount_years``, zero after termination) with ``cand_return_available``
  [H] (reached, or terminated earlier so the return is final);
* independent re-evaluation (h12-dither streams 1..S at
  ``root_final_horizon``, common random numbers across candidates of a root):
  every candidate that survives to the final horizon is re-evaluated, so every
  same-root pair has paired streams and a standard error.
  ``cand_reeval_available``, ``cand_reeval_mask`` [S] (stream evaluated),
  ``cand_reeval_returns`` [S, 4], ``cand_reeval_horizon`` [S] (decisions
  reached per stream), ``cand_reeval_terminated`` [S], ``cand_reeval_mean`` [4] and
  ``cand_reeval_se`` [4] (sample std over streams, ddof 1, / sqrt S). Paired
  SEs of a difference come from the per-stream rows (same stream index);
* ranker bookkeeping: ``cand_predicted_advantage`` [4] (A per
  :data:`COMPONENTS`, normalized units) and ``cand_predicted_available``.

Returns are in raw component units (QoL-years, population-years, feasibility
cost, settlement guard) in :data:`COMPONENTS` order; divide by the frozen
reference scales recorded in ``candidate_payload.reference_scales`` for
normalized G. Pairs must be formed within one root, at one horizon and on one
stream; never across roots, horizons or streams.

:func:`load_search_payload` verifies the bindings and returns the two tables
with global root indices. Like every training loader it refuses held-out-test
outputs and reserved worlds unless ``training=False``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from . import contracts
from .labor_bridge import ACTIVITIES, SPECIALTIES

PAYLOAD_SCHEMA = "labor-search-candidate-payload-v1"
PAYLOAD_SCHEMA_VERSION = 1
ROOTS_DIR = "advantage-roots"
CANDIDATES_DIR = "advantage-candidates"
COMPONENTS = ("qol_years", "population_years", "feasibility", "settlement_guard")
HEADS = ("labor", "movement", "fission", "trade_gate", "trade_value", "trade_share")
ROOT_SOURCE_CODES = {"h12": 0, "actor": 1}
RANKER_OFF, RANKER_PRUNED, RANKER_FULL = 0, 1, 2
# The actor's movement union (labor_model, schema 12): no-op, seven site rows,
# keep the destination intent, eight region rows.
MOVEMENT_UNION_CODES = (
    -1,
    *range(contracts.MOVEMENT_SLOTS - 1),
    contracts.KEEP_INTENT,
    *(contracts.REGION_BASE + row for row in range(contracts.REGION_ROWS)),
)
ROLES = len(SPECIALTIES) - 1
TRADE_GROUPS = len(contracts.TRADE_GROUPS)
ACTION_KEYS = (
    "action_specialty_allocation",
    "action_activity_allocation",
    "action_movement_code",
    "action_movement_slot",
    "action_fission",
    "action_trade_gate",
    "action_trade_value",
    "action_trade_share",
    "action_specialty_counts",
    "action_worker_years",
    "action_population",
    "action_turn_years",
)


def movement_slot(code: int) -> int:
    try:
        return MOVEMENT_UNION_CODES.index(int(code))
    except ValueError as error:
        raise ValueError(f"movement code {code} is not in the schema-12 movement union") from error


def action_arrays(
    payload: Mapping[str, Any], counts: Sequence[int], turn_years: float
) -> dict[str, np.ndarray]:
    """The complete schema-12 action of one candidate, raw and compiled (no batch axis)."""
    specialty = np.asarray(payload["laborSpecialtyAllocation"], dtype=np.float64)
    activity = np.asarray(payload["laborActivityAllocation"], dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    if specialty.shape != (len(SPECIALTIES),) or activity.shape != (ROLES, len(ACTIVITIES)):
        raise ValueError("candidate labor allocation has the wrong shape")
    if counts.shape != (len(SPECIALTIES),) or (counts < 0).any():
        raise ValueError("candidate compiled residents have the wrong shape")
    gate = np.asarray(payload["tradeGate"], dtype=np.int64)
    value = np.asarray(payload["tradeValue"], dtype=np.float64)
    if gate.shape != (TRADE_GROUPS,) or value.shape != (TRADE_GROUPS,):
        raise ValueError("candidate trade posture has the wrong shape")
    years = float(turn_years)
    return {
        "action_specialty_allocation": specialty,
        "action_activity_allocation": activity,
        "action_movement_code": np.asarray(int(payload["movement"]), dtype=np.int64),
        "action_movement_slot": np.asarray(movement_slot(payload["movement"]), dtype=np.int64),
        "action_fission": np.asarray(int(payload["fission"]), dtype=np.int64),
        "action_trade_gate": gate,
        "action_trade_value": value,
        "action_trade_share": np.asarray(float(payload["tradeShare"]), dtype=np.float64),
        "action_specialty_counts": counts,
        "action_worker_years": counts[1:, None].astype(np.float64) * years * activity,
        "action_population": np.asarray(int(counts.sum()), dtype=np.int64),
        "action_turn_years": np.asarray(years, dtype=np.float64),
    }


def returns_by_horizon(
    steps: Sequence[Sequence[float]],
    years: Sequence[float],
    horizons: Sequence[int],
    *,
    terminated: bool,
    discount_years: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative discounted return at each declared horizon, and its availability.

    The discount after step t is ``exp(-years_t / discount_years)`` (physical
    time, as the search branches accumulate it). A horizon beyond the reached
    decisions is available only if the focal camp terminated (its return is
    then final: zero reward after termination).
    """
    steps = np.asarray(steps, dtype=np.float64).reshape(-1, len(COMPONENTS))
    if len(years) != len(steps):
        raise ValueError("per-step years and rewards differ in length")
    out = np.zeros((len(horizons), len(COMPONENTS)), dtype=np.float64)
    available = np.zeros(len(horizons), dtype=np.bool_)
    running = np.zeros(len(COMPONENTS), dtype=np.float64)
    discount = 1.0
    cumulative = [running.copy()]
    for reward, dt in zip(steps, years, strict=True):
        running = running + discount * reward
        discount *= math.exp(-float(dt) / float(discount_years))
        cumulative.append(running.copy())
    reached = len(steps)
    for k, h in enumerate(horizons):
        h = int(h)
        if h <= reached:
            out[k] = cumulative[h]
            available[k] = True
        elif terminated:
            out[k] = cumulative[reached]
            available[k] = True
    return out, available


def reeval_summary(
    streams: Mapping[int, Mapping[str, Any]], count: int
) -> dict[str, np.ndarray]:
    """Per-stream returns (streams 1..count) and the mean/SE over streams."""
    returns = np.zeros((count, len(COMPONENTS)), dtype=np.float64)
    horizon = np.zeros(count, dtype=np.int64)
    terminated = np.zeros(count, dtype=np.bool_)
    mask = np.zeros(count, dtype=np.bool_)
    for stream, entry in streams.items():
        index = int(stream) - 1
        if not 0 <= index < count:
            raise ValueError(f"re-evaluation stream {stream} outside 1..{count}")
        returns[index] = np.asarray(entry["returns"], dtype=np.float64)
        horizon[index] = int(entry["horizon"])
        terminated[index] = bool(entry["terminated"])
        mask[index] = True
    available = bool(mask.all()) and count >= 2
    mean = returns.mean(axis=0) if available else np.zeros(len(COMPONENTS))
    se = (
        returns.std(axis=0, ddof=1) / math.sqrt(count)
        if available
        else np.zeros(len(COMPONENTS))
    )
    return {
        "cand_reeval_available": np.asarray(available, dtype=np.bool_),
        "cand_reeval_returns": returns,
        "cand_reeval_horizon": horizon,
        "cand_reeval_terminated": terminated,
        "cand_reeval_mask": mask,
        "cand_reeval_mean": mean,
        "cand_reeval_se": se,
    }


def digest_bytes(hex_digest: str) -> np.ndarray:
    raw = bytes.fromhex(hex_digest)
    if len(raw) != 32:
        raise ValueError("candidate digest must be sha256")
    return np.frombuffer(raw, dtype=np.uint8).copy()


def stack_rows(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, np.ndarray]:
    """Stack per-row arrays (no batch axis) into columns; every row needs every key."""
    if not rows:
        raise ValueError(f"{label} has no rows")
    keys = set(rows[0])
    for row in rows[1:]:
        if set(row) != keys:
            raise ValueError(f"{label} rows have different columns: {sorted(keys ^ set(row))}")
    out = {}
    for key in sorted(keys):
        values = [_numpy(row[key]) for row in rows]
        out[key] = np.stack(values)
    return out


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def root_row_from_tensors(tensors: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Drop the batch axis of a one-row tensor dict (actor-visible + privileged)."""
    out = {}
    for key, value in tensors.items():
        array = _numpy(value)
        if array.shape[:1] != (1,):
            raise ValueError(f"root tensor {key} is not a single row")
        out[key] = array[0]
    return out


# ---------------------------------------------------------------------------
# Shard binding, verification and loading
# ---------------------------------------------------------------------------


def write_payload_tables(
    staging: Path,
    roots: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    world: int,
    source_run: str,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Write both tables under ``staging`` (which must not exist); returns the meta binding."""
    from .labor_teacher_data import checkpoint_hash, write_shard

    root_arrays = stack_rows(roots, "search payload roots")
    cand_arrays = stack_rows(candidates, "search payload candidates")
    n_roots = len(root_arrays["root_index"])
    if not np.array_equal(root_arrays["root_index"], np.arange(n_roots)):
        raise ValueError("search payload root indices must be 0..R-1 in order")
    if not np.isin(cand_arrays["cand_root"], root_arrays["root_index"]).all():
        raise ValueError("search payload candidate refers to an unknown root")
    staging.mkdir(parents=True)
    tables = {}
    for name, arrays in ((ROOTS_DIR, root_arrays), (CANDIDATES_DIR, cand_arrays)):
        table = write_shard(
            staging / name,
            arrays,
            {
                "payload_schema": PAYLOAD_SCHEMA,
                "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
                "table": name,
                "world_seed": int(world),
                "source_run": source_run,
            },
        )
        tables[name] = {
            "rows": int(table["rows"]),
            "data_digest": table["data_digest"],
            "meta_sha256": checkpoint_hash(staging / name / "meta.json"),
        }
    return {
        "schema": PAYLOAD_SCHEMA,
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "tables": tables,
        **dict(record),
    }


def verify_payload_tables(shard: Path, meta: Mapping[str, Any]) -> dict[str, Any]:
    """Check the bound sub-tables of one search shard against its meta."""
    from .labor_teacher_data import _payload, checkpoint_hash
    from .provenance import ProvenanceError, digest

    binding = meta.get("candidate_payload")
    if not isinstance(binding, Mapping):
        raise ProvenanceError(f"search shard {shard.name} has no candidate payload")
    if (
        binding.get("schema") != PAYLOAD_SCHEMA
        or binding.get("schema_version") != PAYLOAD_SCHEMA_VERSION
    ):
        raise ProvenanceError(
            f"search payload schema {binding.get('schema')!r} != supported {PAYLOAD_SCHEMA}"
        )
    for name in (ROOTS_DIR, CANDIDATES_DIR):
        entry = binding["tables"][name]
        directory = shard / name
        if checkpoint_hash(directory / "meta.json") != entry["meta_sha256"]:
            raise ProvenanceError(f"search payload {name} meta hash mismatch")
        table_meta = json.loads((directory / "meta.json").read_text())
        actual = _payload(directory)
        if (
            actual != table_meta.get("payload")
            or digest(actual) != table_meta.get("data_digest")
            or table_meta["data_digest"] != entry["data_digest"]
            or table_meta.get("source_run") != meta.get("source_run")
        ):
            raise ProvenanceError(f"search payload {name} payload hash mismatch")
    return dict(binding)


def load_search_payload(
    directory: Path, *, training: bool = True, drop_reserved: bool = False
) -> dict[str, Any]:
    """Verified roots and candidates of a search dataset, with global root indices.

    Returns ``{"roots": {key: tensor}, "candidates": {key: tensor},
    "collection": {...}}``; ``candidates["cand_root"]`` indexes the rows of
    ``roots``. Training readers (the default) refuse held-out-test data and
    reserved worlds 90154-90159, like every labor training loader.
    """
    from .labor_teacher_data import _training_view, verify_dataset
    from .provenance import ProvenanceError

    directory = Path(directory)
    manifest = verify_dataset(directory)
    manifest = _training_view(manifest, directory, training=training, drop=drop_reserved)
    roots: dict[str, list[np.ndarray]] = {}
    cands: dict[str, list[np.ndarray]] = {}
    offset = 0
    bindings = {}
    for world, entry in sorted(manifest["shards"].items(), key=lambda kv: int(kv[0])):
        shard = directory / entry["path"]
        meta = json.loads((shard / "meta.json").read_text())
        bindings[str(world)] = verify_payload_tables(shard, meta)
        root_part = _load_table(shard / ROOTS_DIR)
        cand_part = _load_table(shard / CANDIDATES_DIR)
        if set(root_part["row_world"].tolist()) != {int(world)}:
            raise ProvenanceError("search payload roots belong to another world")
        cand_part["cand_root"] = cand_part["cand_root"] + offset
        root_part["root_index"] = root_part["root_index"] + offset
        offset += len(root_part["root_index"])
        for target, part in ((roots, root_part), (cands, cand_part)):
            if target and set(target) != set(part):
                # Privileged arrays exist only when the producer emits them.
                raise ProvenanceError("search payload shards store different columns")
            for key, value in part.items():
                target.setdefault(key, []).append(value)
    if not roots:
        raise ProvenanceError("search dataset has no candidate payload")
    first = next(iter(bindings.values()))
    return {
        "roots": {k: torch.from_numpy(np.concatenate(v)) for k, v in roots.items()},
        "candidates": {k: torch.from_numpy(np.concatenate(v)) for k, v in cands.items()},
        "collection": {
            "payload_schema": PAYLOAD_SCHEMA,
            "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
            "dataset_digest": manifest["dataset_digest"],
            "source_run": manifest["source_run"],
            "world_seeds": sorted(int(w) for w in manifest["shards"]),
            "components": list(COMPONENTS),
            "reference_scales": first.get("reference_scales"),
            "discount_years": first.get("discount_years"),
            "horizons": first.get("horizons"),
            "partition": manifest["source_record"].get("options", {}).get("partition"),
            "dropped_reserved_worlds": manifest.get("dropped_reserved_worlds", []),
        },
    }


def _load_table(directory: Path) -> dict[str, np.ndarray]:
    meta = json.loads((directory / "meta.json").read_text())
    return {
        name[: -len(".npy")]: np.load(directory / name, allow_pickle=False)
        for name in sorted(meta["payload"])
    }


# ---------------------------------------------------------------------------
# A-ranker hook (labor_advantage, model side; off by default)
# ---------------------------------------------------------------------------


class CandidateRanker(Protocol):
    """What ``labor_search`` needs from a trained advantage head.

    ``score`` receives the root row (``advantage-roots`` columns as one-row
    tensors, actor-visible and privileged) and the candidates' action columns
    (:data:`ACTION_KEYS`, one row per candidate) and returns ``[n, 4]``
    advantages per :data:`COMPONENTS` in normalized units (return / frozen
    reference scale). Centering over the candidate set is applied by the
    search; the ranker may return uncentered values.
    """

    def score(
        self, root: Mapping[str, torch.Tensor], candidates: Mapping[str, torch.Tensor]
    ) -> torch.Tensor: ...


def load_ranker(path: Path) -> CandidateRanker:
    """The advantage-head ranker from ``labor_advantage.load_search_ranker`` (model side)."""
    try:
        from . import labor_advantage  # type: ignore[attr-defined]
    except ImportError as error:  # pragma: no cover - depends on the model branch
        raise RuntimeError(
            "the A-ranker needs labor_advantage.load_search_ranker (feature/r5-advantage)"
        ) from error
    loader = getattr(labor_advantage, "load_search_ranker", None)
    if loader is None:
        raise RuntimeError("labor_advantage provides no load_search_ranker(path)")
    return loader(Path(path))


def centered_scores(scores: Any, n: int) -> np.ndarray:
    array = _numpy(scores).astype(np.float64)
    if array.shape != (n, len(COMPONENTS)) or not np.isfinite(array).all():
        raise ValueError(f"ranker scores must be finite [{n}, {len(COMPONENTS)}]")
    return array - array.mean(axis=0, keepdims=True)


def full_evaluation_root(key: Sequence[int], fraction: float) -> bool:
    """Keyed draw: does this root keep the full native evaluation (regret measurement)?"""
    if not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("full-evaluation fraction must be in [0, 1]")
    seed = int.from_bytes(
        hashlib.sha256(json.dumps([int(k) for k in key]).encode()).digest()[:8], "little"
    )
    return bool(np.random.default_rng(seed).random() < float(fraction))
