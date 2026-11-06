"""Critic bias on search branch states (docs/MODEL9_SEARCH_TEACHER.md, *Validation*).

    python -m anthropocene_rl.autonomous_v7.branch_bias --checkpoint <pt> --branch-shards <dir>[,<dir>] [--out bias.json]

On held-out branch shards (search_teacher ``<out>/branch_shards`` from worlds the critic never trained on), per
alternative kind and objective head, as relative bias (prediction - realised) / |realised|, mean and median:

* ``calibration`` (the gate): calibration branches (12-turn tails). Per branch, the critic's implied 8-turn lineage
  value V(s_0) - Gamma_8 V(s_8) against the realised discounted 8-turn lineage return, with no bootstrap in the
  target. ``pass``: within 10% (mean and median, welfare head) for native, region and mix.
* ``coverage``: evaluable calibration branches, including zero-return targets, per kind/objective and population
  band. Held-out manifests, sufficient mandatory welfare coverage, independent near-zero/relative-error gates and
  declared nonmandatory waivers are required for a passing verdict (MODEL9_ROUND3_BUILD_ARCHITECTURE.md).
* ``saturation`` (round 3): per objective head, the share of rows whose value expectation sits within 1% of the
  two-hot bin range's ends (a saturated head cannot express larger or smaller values).
* ``self_consistency`` (diagnostic): every branch; the critic's value against the tail return plus its own
  discounted bootstrap after the last tail turn (lambda = 1). With a 4-turn tail most of that target is the
  bootstrap, so it measures consistency, not calibration. ``entry`` is one number per branch (lineage sums at the
  first tail turn), ``rows`` every tail row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import branch_shards, data
from .model import load_checkpoint_normalizer
from .search_teacher import KINDS, MIX, NATIVE, REGION, load_critic
from .upgrade import upgrade

TARGET = 0.10
GATED = (NATIVE, REGION, MIX)
CALIBRATION_TURNS = 8
# Coverage minimums (review 5.5(3)): calibration branches per kind x objective cell, and per kind x population band.
MIN_CELL = 20
MIN_BAND = 5
# Lineage population at the fork's first tail turn: [lower, upper) bounds of each band.
POPULATION_BANDS = ((0, 30), (30, 100), (100, 300), (300, 10**9))
SATURATION_EDGE = 0.01


def relative(pred: np.ndarray, real: np.ndarray) -> np.ndarray:
    return (pred - real) / np.maximum(1e-9, np.abs(real))


def realised(
    critic, rows: dict[str, np.ndarray], boot, normalizer
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Predicted raw values per head and the realised tail return plus bootstrap (lambda = 1)."""
    value, bootstrap = branch_shards.values(critic, rows, boot, normalizer)
    components = data.reward_components(rows)
    real = {
        name: data.lambda_returns(rows, components[name], value[name], 1.0, bootstrap[name])
        for name in value
    }
    return value, real


def report(
    pred: dict[str, np.ndarray], real: dict[str, np.ndarray], rows: dict[str, np.ndarray]
) -> dict:
    kinds = rows["alt_kind"].astype(int)
    world, turn = rows["world"], rows["turn"]
    # Entry rows: each branch world's first tail turn (all lineage settlements deciding then).
    first = {}
    for w, t in zip(world.tolist(), turn.tolist(), strict=False):
        first[w] = min(first.get(w, t), t)
    entry = np.asarray([first[w] == t for w, t in zip(world.tolist(), turn.tolist(), strict=False)])
    out: dict = {
        "rows": len(kinds),
        "branches": len(first),
        "target": TARGET,
        "kinds": {},
    }
    for k in sorted(set(kinds.tolist())):
        sel = kinds == k
        by_head = {}
        for name in pred:
            r = relative(pred[name][sel], real[name][sel])
            worlds = sorted(set(world[sel & entry].tolist()))
            e = np.asarray(
                [
                    relative(
                        np.asarray([pred[name][(world == w) & entry].sum()]),
                        np.asarray([real[name][(world == w) & entry].sum()]),
                    )[0]
                    for w in worlds
                ]
            )
            by_head[name] = {
                "entry_mean": float(np.mean(e)) if len(e) else None,
                "entry_median": float(np.median(e)) if len(e) else None,
                "rows_mean": float(np.mean(r)),
                "rows_median": float(np.median(r)),
                "branches": len(worlds),
                "rows": int(sel.sum()),
            }
        out["kinds"][KINDS[k] if 0 <= k < len(KINDS) else str(k)] = by_head
    return out


from .calibration import calibration, coverage, held_out, manifest, verdict


def saturation(u: dict[str, np.ndarray], edge: float = SATURATION_EDGE) -> dict[str, float]:
    """Per head, the share of value expectations (in the head's own two-hot units: per resident, qol_years raw) whose
    symlog lies within ``edge`` of the bin range's ends."""
    from .model import BIN_HIGH, BIN_LOW

    margin = edge * (BIN_HIGH - BIN_LOW)
    out = {}
    for name, x in u.items():
        y = np.sign(x) * np.log1p(np.abs(x))
        out[name] = (
            float(np.mean((y >= BIN_HIGH - margin) | (y <= BIN_LOW + margin))) if len(y) else 0.0
        )
    return out


def evaluate(
    pred: dict[str, np.ndarray],
    real: dict[str, np.ndarray],
    rows: dict[str, np.ndarray],
    endpoints: dict | None = None,
    manifest_verified: bool = False,
    waivers=(),
) -> dict:
    cal = calibration(pred, rows, endpoints=endpoints)
    cov = coverage(rows)
    gate = verdict(cal, cov, waivers)
    if not manifest_verified:
        gate["pass"] = False
        gate["failures"].append("held-out calibration manifest not verified")
    return {
        **gate,
        "calibration": cal,
        "coverage": cov,
        "self_consistency": report(pred, real, rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--branch-shards", required=True, help="comma-separated held-out branch-shard directories"
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--provenance",
        choices=("reject", "flag", "allow"),
        default="flag",
        help="held-out shards that mix sources or hold shards of unknown source (default: flag)",
    )
    parser.add_argument("--calibration-manifest", type=Path, default=None)
    parser.add_argument(
        "--calibration-inputs",
        type=Path,
        default=None,
        help="full frozen calibration cohort/branch payload inventory",
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    state = upgrade(torch.load(args.checkpoint, map_location="cpu", weights_only=False))
    held = manifest(args.calibration_manifest) if args.calibration_manifest else None
    if held:
        from .lineage import checkpoint_evidence
        from .provenance import ProvenanceError, checkpoint_hash

        if (
            held.get("waivers", []) != []
            or state.get("artifact_kind") != "promotion"
            or not state.get("vector_critic")
            or not checkpoint_evidence(state, held)["verified"]
        ):
            raise ProvenanceError(
                "one-shot Round-3 calibration requires an already-promoted four-head candidate and no waivers"
            )
        if args.provenance != "reject" or args.out is None:
            raise ProvenanceError(
                "qualification calibration requires reject provenance and a new output path"
            )
        if args.calibration_inputs is None:
            raise ProvenanceError("calibration requires exact frozen --calibration-inputs")
        from .calibration_audit import claim, inventory

        binding = inventory(
            [Path(p) for p in args.branch_shards.split(",")],
            checkpoint_hash(args.checkpoint),
            held,
            json.loads(args.calibration_inputs.read_text()),
        )
        prechecked = {}
        preloaded = branch_shards.load(
            [Path(p) for p in args.branch_shards.split(",")],
            "reject",
            prechecked,
            purpose="analysis",
        )
        audit_claim = claim(held, binding)
    normalizer = load_checkpoint_normalizer(state)
    critic = load_critic(args.checkpoint)
    provenance: dict = {}
    if held:
        rows, boot = preloaded
        provenance.update(prechecked)
    else:
        rows, boot = branch_shards.load(
            [Path(p) for p in args.branch_shards.split(",")],
            args.provenance,
            provenance,
            purpose="analysis",
        )
    pred, real = realised(critic, rows, boot, normalizer)
    endpoints = branch_shards.endpoint_values(critic, boot, normalizer)
    from .lineage import checkpoint_evidence
    from .provenance import ProvenanceError, compatible_sources

    checkpoint_check = (
        checkpoint_evidence(state, held)
        if held
        else {"verified": False, "failures": ["no exclusion manifest"]}
    )
    source_check = {"compatible": False}
    if checkpoint_check["verified"]:
        try:
            frozen = state["run_manifest"]
            source_check = compatible_sources(
                provenance.get("records", []),
                frozen["simulator"],
                frozen.get("simulator_compatibility", []),
                {k: frozen[k] for k in ("sim_rules", "sim_statics")},
            )
        except ProvenanceError as error:
            source_check = {"compatible": False, "failure": str(error)}
    verified = (
        bool(held)
        and checkpoint_check["verified"]
        and source_check["compatible"]
        and provenance["identity_valid"]
        and provenance["ok"]
        and all(
            held_out(held, int(m["seed"]), m.get("snapshot_hash"))
            and m.get("calibration_manifest_digest") == held["manifest_digest"]
            for m in provenance["branches"]
        )
    )
    u = {
        name: v / branch_shards.head_scale(name, rows["prior_population"])
        for name, v in pred.items()
    }
    result = {
        "checkpoint": str(args.checkpoint),
        "provenance": provenance,
        "audit_binding": binding if held else None,
        "audit_claim": audit_claim if held else None,
        "calibration_inputs": str(args.calibration_inputs) if args.calibration_inputs else None,
        "checkpoint_evidence": checkpoint_check,
        "source_compatibility": source_check,
        **evaluate(pred, real, rows, endpoints, verified, (held or {}).get("waivers", [])),
        "saturation": saturation(u),
    }
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
