"""Search candidate-payload export for the advantage head (intervention B training data).

The advantage head ``A_j(C, a)`` (``labor_advantage``, model side, branch
``feature/r5-advantage``) is trained on within-root return differences between
complete candidate actions restored from one identical root. The label rows of
``labor_search`` keep only the chosen label per personality; this module writes
*every* evaluated candidate in the model side's consumer contract,
``labor-advantage-candidates-v1`` (``labor_advantage.CANDIDATE_PAYLOAD_SCHEMA``;
its module docstring is the normative field list and ``parse_candidate_shard``
its strict reader). The search exporter conforms to that schema and adds the
producer fields below under keys the reader ignores; the extension is
versioned as :data:`EXPORT_SCHEMA`.

Layout. Each search shard ``world-<seed>/`` stores one ``torch.save`` dict,
:data:`PAYLOAD_FILE`, hash-bound in the shard meta under
``candidate_payload`` (schema, file, sha256, root and candidate counts), so
``load_candidate_shards([<shard dir>, ...])`` reads it directly and
:func:`candidate_shard_paths` returns the verified paths of a published search
dataset (refusing held-out-test data and reserved worlds for training).

Shard (``labor-advantage-candidates-v1`` fields, then exporter extensions):

* ``schema``, ``component_order`` (qol_years, population_years, feasibility,
  settlement_guard), ``returns_units`` (raw discounted component return),
  ``discount_years`` (400), ``specialty_ids`` [14], ``activity_ids`` [22],
  ``trade_groups`` [7];
* ``tensors``: the focal camp's actor-visible observation rows at each root
  exactly as ``labor_policy_tensors`` builds them, ``labor_personality`` set
  to the root personality (no privileged key); ``privileged``: the raw
  critic-only arrays (``labor_privileged.shard_keys``) of the same rows, or
  None for a pre-privileged producer;
* ``producer``: exporter identity (:data:`EXPORT_SCHEMA`), source run, world,
  partition, root source, the declared search configuration, reference scales,
  family codes, the movement-union codes and the ranker binding;
* ``roots`` -- per root: ``root_id``, ``world``, ``turn``, ``camp``,
  ``society``, ``source`` (``h12``/``actor``: whose trajectory the root is
  on), ``tensor_row``, ``personality``, ``full_evaluation`` (no candidate was
  skipped by the A-ranker; successive halving still prunes by horizon),
  ``pair_sd`` (None: pair SEs come from the re-evaluation streams); extensions
  ``final_horizon`` (decisions reached by the re-evaluated survivors),
  ``truncated`` (the resolve cap stopped halving), ``horizons`` (declared),
  ``reeval_streams``, ``ranker_mode`` (0 off, 1 pruned to top-k plus H12,
  2 ranked but fully evaluated), ``population``, ``turn_years``;
* per candidate: ``index``, ``family``, ``is_h12``, ``digest``
  (``labor_search.payload_digest``), ``action`` (the complete executed schema-12
  payload in raw output units), ``control`` (``all-idle`` when every resident
  is idle, else None), ``bookkeeping_twin`` (None) and ``evaluations``; the
  extensions ``description``, ``chosen_heads``, ``attempt_valid``,
  ``natively_evaluated``, ``compiled`` (native largest-remainder residents
  ``specialty_counts`` [14], ``worker_years`` [13][22] = residents x share x
  ``turn_years``, ``population``, ``turn_years``, ``movement_slot`` in the
  actor's 17-way movement union), ``horizon_reached``, ``terminated``,
  ``last_round``, ``step_rewards`` [[4]] and ``step_years`` of the selection
  branch, and ``predicted_advantage`` (centred A per component from the
  ranker, or None);
* evaluations: stream 0 (the deterministic H12 continuation every candidate
  shares) at every declared horizon the branch reached or terminated before,
  and streams 1..S (independent h12-dither re-evaluation with common random
  numbers across a root's candidates) at the final horizon for every candidate
  that survived to it, so every same-root survivor pair has paired streams.
  Each: ``horizon``, ``reached_horizon``, ``terminated``, ``stream``,
  ``returns`` [4], ``step_years``, ``se`` (None; paired SEs come from shared
  streams).
* T2 plan rows (``--plan-candidates N``, after every first-action candidate):
  per plan a ``plan`` row (family ``plan``: the focal camp holds the plan's
  course for ``plan_length`` decisions, then H12) and its paired
  ``first-step-control`` row (family ``plan-control``: the same first step, then
  H12 from decision 2). Both carry the shared first step as ``action`` and are
  evaluated on stream 0 at every declared horizon up to the root's
  ``plan_horizon`` = max(final horizon, P + 1) and on streams 1..S at
  ``plan_horizon``, with common random numbers shared with each other and with
  the root's candidates. Fields: ``plan_id``, ``plan_role`` (``plan`` /
  ``first-step-control``), ``plan_template``, ``plan_length``, ``plan_steps``
  (stream 0: per step ``step``, ``action``, ``digest``, ``description``,
  ``valid``, ``fallback`` (None, or ``h12:<reason>`` when the step was not
  constructible or natively rejected and H12 ran), ``conserves``,
  ``specialty_counts``, ``population``, ``same_as_h12``; the control lists only
  step 0), ``plan_continuation`` (``h12``), ``plan_step_digests`` and
  ``plan_fallbacks`` per stream, ``step_returns`` and ``step_years_by_stream``
  ({stream: per-step raw rewards [4] / years}). Roots record
  ``plan_candidates``, ``plan_steps``, ``plan_horizon`` and ``plan_unavailable``
  (templates not constructible at the root). First-action readers
  (``labor_advantage.parse_candidate_shard``) keep plan rows apart in
  ``Root.plans``.

Successor rows (export v2, :data:`EXPORT_SCHEMA_SUCCESSORS`; ``--successor-rows``).
Additive: every v1 field keeps its value and readers that ignore unknown keys
(``labor_advantage.parse_candidate_shard``, ``labor_r5_acceptance``) read v2
shards unchanged. Without the flag the export is v1, byte for byte.

* shard: ``successor_tensors`` (the focal camp's actor-visible observation rows at
  the successor states, exactly the root-row format: ``labor_policy_tensors``
  through the run's normalizer, ``labor_personality`` set to the root
  personality), ``successor_privileged`` (the raw critic-only arrays of the
  same rows, or None for a pre-privileged producer) and ``successor_row_count``.
  Identical rows within one root are stored once (content digest) and shared
  by index;
* root: ``successor_decisions`` (the decision boundaries that carry rows: 1 and
  the last declared horizon) and ``successor_rows`` (rows this root added);
* first-action candidate: ``successor_streams``, one entry per evaluated stream
  (0 when natively evaluated; 1..S when re-evaluated): ``stream``,
  ``step_rewards`` [[4]] and ``step_years`` (raw per-step rewards of that
  stream's branch), ``terminated``, and ``boundaries``: per decision boundary d
  reached, ``decision``, ``turn`` (native turn of the successor state),
  ``status`` (``live``: the focal camp has a labor observation there;
  ``terminated``; ``inactive``: present without an observation or labor
  catalog, the branch ends) and ``row`` (index into ``successor_tensors`` for a
  live boundary in ``successor_decisions``, else None). Plan rows carry no
  successor streams.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import contracts
from .labor_bridge import ACTIVITIES, SPECIALTIES

# Must equal labor_advantage.CANDIDATE_PAYLOAD_SCHEMA / RETURNS_UNITS (model side).
CANDIDATE_PAYLOAD_SCHEMA = "labor-advantage-candidates-v1"
RETURNS_UNITS = "raw-discounted-component-return"
EXPORT_SCHEMA = "labor-search-candidate-export-v1"
# v2 = v1 plus successor critic-input rows (additive; see the module docstring).
EXPORT_SCHEMA_SUCCESSORS = "labor-search-candidate-export-v2"
EXPORT_SCHEMAS = (EXPORT_SCHEMA, EXPORT_SCHEMA_SUCCESSORS)
SUCCESSOR_STATUSES = ("live", "terminated", "inactive")
PAYLOAD_FILE = "advantage-candidates.pt"
COMPONENTS = ("qol_years", "population_years", "feasibility", "settlement_guard")
HEADS = ("labor", "movement", "fission", "trade_gate", "trade_value", "trade_share")
ROOT_SOURCES = ("h12", "actor")
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


def movement_slot(code: int) -> int:
    try:
        return MOVEMENT_UNION_CODES.index(int(code))
    except ValueError as error:
        raise ValueError(f"movement code {code} is not in the schema-12 movement union") from error


def compiled_action(
    payload: Mapping[str, Any], counts: Sequence[int], turn_years: float
) -> dict[str, Any]:
    """The candidate's action compiled to native units (residents and worker-years)."""
    activity = np.asarray(payload["laborActivityAllocation"], dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    if activity.shape != (ROLES, len(ACTIVITIES)) or counts.shape != (len(SPECIALTIES),):
        raise ValueError("candidate labor allocation has the wrong shape")
    if (counts < 0).any():
        raise ValueError("candidate compiled residents must be nonnegative")
    years = float(turn_years)
    return {
        "specialty_counts": [int(c) for c in counts],
        "worker_years": (counts[1:, None].astype(np.float64) * years * activity).tolist(),
        "population": int(counts.sum()),
        "turn_years": years,
        "movement_slot": movement_slot(payload["movement"]),
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


def selection_evaluations(
    steps: Sequence[Sequence[float]],
    years: Sequence[float],
    horizons: Sequence[int],
    *,
    terminated: bool,
    discount_years: float,
) -> list[dict[str, Any]]:
    """Stream-0 evaluations at every declared horizon the branch reached (or ended before)."""
    by_h, available = returns_by_horizon(
        steps, years, horizons, terminated=terminated, discount_years=discount_years
    )
    reached = len(steps)
    out = []
    for k, h in enumerate(horizons):
        if not available[k]:
            continue
        resolved = min(int(h), reached)
        out.append(
            {
                "horizon": int(h),
                "reached_horizon": resolved,
                # The focal camp ended inside this window (at or before decision h).
                "terminated": bool(terminated and reached <= int(h)),
                "stream": 0,
                "returns": [float(x) for x in by_h[k]],
                "step_years": [float(y) for y in years[:resolved]],
                "se": None,
            }
        )
    return out


def reevaluation_records(
    streams: Mapping[int, Mapping[str, Any]], horizon: int
) -> list[dict[str, Any]]:
    """Streams >= 1 at the final horizon (complete or terminated branches only)."""
    out = []
    for stream in sorted(streams):
        entry = streams[stream]
        reached = int(entry["horizon"])
        terminated = bool(entry["terminated"])
        if int(stream) < 1:
            raise ValueError("re-evaluation streams start at 1")
        out.append(
            {
                "horizon": int(horizon),
                "reached_horizon": reached,
                "terminated": terminated,
                "stream": int(stream),
                "returns": [float(x) for x in np.asarray(entry["returns"], dtype=np.float64)],
                "step_years": [float(y) for y in entry.get("years", [])],
                "se": None,
            }
        )
    return out


def export_schema(successor_rows: bool) -> str:
    return EXPORT_SCHEMA_SUCCESSORS if successor_rows else EXPORT_SCHEMA


def _row_digest(row: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for part in ("tensors", "privileged"):
        values = row.get(part)
        digest.update(part.encode())
        if values is None:
            digest.update(b"none")
            continue
        for key in sorted(values):
            array = torch.as_tensor(values[key]).detach().cpu().contiguous()
            digest.update(key.encode())
            digest.update(str(array.dtype).encode())
            digest.update(repr(tuple(array.shape)).encode())
            digest.update(array.numpy().tobytes())
    return digest.hexdigest()


class SuccessorRows:
    """One root's successor critic-input rows, deduplicated by content.

    ``add`` takes ``{"tensors": one-row actor-visible tensors, "privileged": one-row
    raw privileged arrays or None}``, sets ``labor_personality`` to the root
    personality (as the root row has it) and returns the root-local row index."""

    def __init__(self, personality: Sequence[float]) -> None:
        self.personality = torch.tensor([[float(x) for x in personality]], dtype=torch.float32)
        self.rows: list[dict[str, Any]] = []
        self._index: dict[str, int] = {}

    def add(self, row: Mapping[str, Any]) -> int:
        tensors = {k: torch.as_tensor(v) for k, v in row["tensors"].items()}
        tensors["labor_personality"] = self.personality.clone()
        privileged = row.get("privileged")
        item = {
            "tensors": tensors,
            "privileged": None
            if privileged is None
            else {k: torch.as_tensor(v) for k, v in privileged.items()},
        }
        key = _row_digest(item)
        if key not in self._index:
            self._index[key] = len(self.rows)
            self.rows.append(item)
        return self._index[key]


def successor_stream(
    stream: int,
    steps: Sequence[Sequence[float]],
    years: Sequence[float],
    terminated: bool,
    boundaries: Sequence[Mapping[str, Any]],
    store: SuccessorRows,
) -> dict[str, Any]:
    """One candidate stream's successor record (root-local row indices)."""
    out = []
    for b in boundaries:
        status = str(b["status"])
        if status not in SUCCESSOR_STATUSES:
            raise ValueError(f"successor status {status!r} is not one of {SUCCESSOR_STATUSES}")
        row = b.get("row")
        if row is not None and status != "live":
            raise ValueError("only a live successor boundary carries a row")
        out.append(
            {
                "decision": int(b["decision"]),
                "turn": int(b["turn"]),
                "status": status,
                "row": None if row is None else store.add(row),
            }
        )
    return {
        "stream": int(stream),
        "step_rewards": [[float(x) for x in r] for r in steps],
        "step_years": [float(y) for y in years],
        "terminated": bool(terminated),
        "boundaries": out,
    }


def _assemble_successors(
    roots: Sequence[Mapping[str, Any]], records: list[dict[str, Any]], privileged_flag: bool
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None, int]:
    """Concatenate per-root successor rows; rewrite root-local row indices to shard rows."""
    from .labor_privileged import is_privileged_key

    rows: list[Mapping[str, Any]] = []
    for root, record in zip(roots, records, strict=True):
        local = list(root["successors"] or [])
        offset = len(rows)
        record["successor_rows"] = len(local)
        for cand in record["candidates"]:
            for entry in cand.get("successor_streams") or []:
                for b in entry["boundaries"]:
                    if b["row"] is not None:
                        if not 0 <= int(b["row"]) < len(local):
                            raise ValueError("successor row index out of range")
                        b["row"] = offset + int(b["row"])
        rows.extend(local)
    if not rows:
        return {}, None, 0
    keys = set(rows[0]["tensors"])
    for row in rows:
        if set(row["tensors"]) != keys:
            raise ValueError("successor rows carry different observation tensors")
        if (row["privileged"] is not None) != privileged_flag:
            raise ValueError("successor rows and root rows disagree on the privileged view")
    leaked = sorted(k for k in keys if is_privileged_key(k))
    if leaked:
        raise ValueError(f"privileged keys in successor observation tensors: {leaked}")
    tensors = {
        k: torch.cat([torch.as_tensor(r["tensors"][k]) for r in rows]) for k in sorted(keys)
    }
    privileged = None
    if privileged_flag:
        names = sorted(rows[0]["privileged"])
        privileged = {
            k: torch.cat([torch.as_tensor(r["privileged"][k]) for r in rows]) for k in names
        }
    return tensors, privileged, len(rows)


def candidate_shard(
    roots: Sequence[Mapping[str, Any]],
    *,
    producer: Mapping[str, Any],
    discount_years: float,
) -> dict[str, Any]:
    """Assemble one ``labor-advantage-candidates-v1`` shard from per-root results.

    Each element of ``roots`` is ``{"record": root dict, "tensors": one-row
    actor-visible tensors, "privileged": one-row raw privileged arrays or
    None}``; ``record["tensor_row"]`` must equal its position.
    """
    from .labor_privileged import is_privileged_key

    if not roots:
        raise ValueError("a candidate shard needs at least one root")
    if float(discount_years) != contracts.HORIZON_YEARS:
        raise ValueError("candidate payload discount must be the critic's 400-year constant")
    keys = set(roots[0]["tensors"])
    flags = {r["privileged"] is not None for r in roots}
    if len(flags) != 1:
        raise ValueError("candidate roots mix privileged and pre-privileged rows")
    for row, root in enumerate(roots):
        if set(root["tensors"]) != keys:
            raise ValueError("candidate roots carry different observation tensors")
        if int(root["record"]["tensor_row"]) != row:
            raise ValueError("candidate root tensor_row must equal its position")
        if sum(bool(c["is_h12"]) for c in root["record"]["candidates"]) != 1:
            raise ValueError("candidate root must have exactly one H12 candidate")
    leaked = sorted(k for k in keys if is_privileged_key(k))
    if leaked:
        raise ValueError(f"privileged keys in candidate observation tensors: {leaked}")
    tensors = {k: torch.cat([torch.as_tensor(r["tensors"][k]) for r in roots]) for k in keys}
    privileged = None
    if flags == {True}:
        names = set(roots[0]["privileged"])
        privileged = {
            k: torch.cat([torch.as_tensor(r["privileged"][k]) for r in roots]) for k in names
        }
    ids = [r["record"]["root_id"] for r in roots]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate root ids repeat within a shard")
    successors = {r.get("successors") is not None for r in roots}
    if len(successors) != 1:
        raise ValueError("candidate roots mix successor-row and v1 exports")
    with_successors = successors == {True}
    records = [json.loads(json.dumps(r["record"], allow_nan=False)) for r in roots]
    shard: dict[str, Any] = {
        "schema": CANDIDATE_PAYLOAD_SCHEMA,
        "producer": dict(producer, export_schema=export_schema(with_successors)),
        "component_order": list(COMPONENTS),
        "returns_units": RETURNS_UNITS,
        "discount_years": float(discount_years),
        "specialty_ids": list(SPECIALTIES),
        "activity_ids": list(ACTIVITIES),
        "trade_groups": list(contracts.TRADE_GROUPS),
        "tensors": tensors,
        "privileged": privileged,
        "roots": records,
    }
    if with_successors:
        s_tensors, s_privileged, count = _assemble_successors(
            roots, records, privileged is not None
        )
        shard["successor_tensors"] = s_tensors
        shard["successor_privileged"] = s_privileged
        shard["successor_row_count"] = count
    return shard


def write_candidate_shard(path: Path, shard: Mapping[str, Any]) -> dict[str, Any]:
    """Save atomically; returns the binding recorded in the search shard meta."""
    from .provenance import checkpoint_hash

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; candidate payloads are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(shard), temporary)
    os.replace(temporary, path)
    roots = shard["roots"]
    binding = {
        "schema": CANDIDATE_PAYLOAD_SCHEMA,
        "export_schema": shard["producer"].get("export_schema", EXPORT_SCHEMA),
        "file": PAYLOAD_FILE,
        "sha256": checkpoint_hash(path),
        "roots": len(roots),
        "candidates": sum(len(r["candidates"]) for r in roots),
        "evaluations": sum(len(c["evaluations"]) for r in roots for c in r["candidates"]),
        "plan_rows": sum(1 for r in roots for c in r["candidates"] if c.get("plan_role")),
    }
    if "successor_row_count" in shard:  # v2 only: v1 bindings are unchanged
        binding["successor_rows"] = int(shard["successor_row_count"])
    return binding


def has_successors(shard: Mapping[str, Any]) -> bool:
    """True for a v2 shard carrying successor rows (``successor_tensors``)."""
    return "successor_row_count" in shard and shard.get("successor_tensors") is not None


def read_candidate_shard(path: Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def verify_candidate_payload(shard: Path, meta: Mapping[str, Any]) -> Path:
    """The bound payload file of one search shard, hash-checked against its meta."""
    from .provenance import ProvenanceError, checkpoint_hash

    binding = meta.get("candidate_payload")
    if not isinstance(binding, Mapping):
        raise ProvenanceError(f"search shard {shard.name} has no candidate payload")
    if binding.get("schema") != CANDIDATE_PAYLOAD_SCHEMA or binding.get("file") != PAYLOAD_FILE:
        raise ProvenanceError(
            f"search payload schema {binding.get('schema')!r} != {CANDIDATE_PAYLOAD_SCHEMA}"
        )
    path = shard / PAYLOAD_FILE
    if not path.is_file() or checkpoint_hash(path) != binding.get("sha256"):
        raise ProvenanceError(f"search payload {path} is missing or its hash differs")
    return path


def candidate_shard_paths(
    directory: Path, *, training: bool = True, drop_reserved: bool = False
) -> list[Path]:
    """Verified payload files of a published search dataset (for ``load_candidate_shards``).

    Training readers (the default) refuse held-out-test data and reserved
    worlds 90154-90159, like every labor training loader; evaluation readers
    pass ``training=False``.
    """
    from .labor_teacher_data import _training_view, verify_dataset

    directory = Path(directory)
    manifest = verify_dataset(directory)
    manifest = _training_view(manifest, directory, training=training, drop=drop_reserved)
    paths = []
    for _, entry in sorted(manifest["shards"].items(), key=lambda kv: int(kv[0])):
        shard = directory / entry["path"]
        meta = json.loads((shard / "meta.json").read_text())
        paths.append(verify_candidate_payload(shard, meta))
    return paths


def digest_text(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# A-ranker hook (labor_advantage.AdvantageRanker, model side; off by default)
# ---------------------------------------------------------------------------


def load_ranker(
    head_path: Path,
    critic_checkpoint: Path,
    *,
    top_k: int,
    full_fraction: float,
    key: int,
) -> Any:
    """``labor_advantage.AdvantageRanker`` over the frozen critic the head was fitted on."""
    try:
        from . import labor_advantage  # type: ignore[attr-defined]
    except ImportError as error:  # pragma: no cover - depends on the model branch
        raise RuntimeError(
            "the A-ranker needs labor_advantage (feature/r5-advantage) on this checkout"
        ) from error
    task, costs = labor_advantage._load_critics(Path(critic_checkpoint))
    head, _ = labor_advantage.load_advantage_head(
        torch.load(Path(head_path), map_location="cpu", weights_only=True), task, costs
    )
    head.eval()
    return labor_advantage.AdvantageRanker(
        task, costs, head, top_k=int(top_k), full_fraction=float(full_fraction), key=int(key)
    )
