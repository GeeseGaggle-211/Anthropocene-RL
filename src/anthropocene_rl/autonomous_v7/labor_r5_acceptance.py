"""r5 critic acceptance harness: enforced stability, fitting acceptance, seeds and receipt.

Frozen design: G0 manifest sections 8-10 (r5 FROZEN 26 Oct 03:15Z), the
intervention brief of 26 Oct and execution brief sections 4 (receipt fields),
9 (G2C) and 17 (stability). This module turns the per-fit evidence into a
machine receipt (``gate-receipt.json``) with verdict PASS / HOLD / FAIL:

* **Stability (mandatory).** Every critic (V) fit and every advantage (A) fit
  must carry an admitted ``labor_stability`` receipt: no abort, rollbacks
  within budget, frozen actor (and, for A, critic) gradients exactly zero,
  no persistent gradient dominance, no QoL saturation, no TD(lambda)
  divergence, a bounded decoded population tail, no A overfit and, for a
  privileged critic, nonzero attention to its privileged tokens. A fit that
  fails a mandatory stability check is not admitted: its gate FAILs whatever
  its calibration numbers are, and a missing receipt HOLDs.
* **QoL (amendment r5-b).** (a) same-village action ranking via A on the
  reserved worlds 90154-90159; (b) bounded-head calibration (nMAE <= 0.20,
  |bias| <= 0.10) per stratum on calibration worlds 90180-90199; (c) TD
  residual consistency along each village's own trajectory.
* **Other heads.** No regression beyond 0.02 against the critic3 control on
  the same calibration worlds. Full G2C for every head is reported as a
  separate gate.
* **A gates** on the reserved worlds: ranking >= 0.70, false improvement
  <= 0.15 at margin 0.01, the all-idle and delayed-benefit controls and the
  bookkeeping probe (``labor_advantage_eval``).
* **Generalisation gap** fit versus calibration per head, and A held-out
  (within fitting worlds) versus reserved.
* **Seeds.** Per-seed checks, seed spread, and the declared seed rule:
  ``all-seeds`` (default: every seed passes every check) or
  ``mean-with-floor`` (the seed mean passes every scalar threshold, the worst
  seed is within ``floor_tolerance`` of it, and every non-scalar and every
  stability check passes in every seed). The receipt records the rule used.
* **Arms.** Privileged critic versus the non-privileged control with the same
  data, support, tail and budget; the paired per-seed effect of privilege is
  reported, and a confounded comparison HOLDs.
* **Determinism.** The same seed and data give bitwise-equal parameters (or
  within the documented CPU tolerance, reported as such).

Subcommands: ``evaluate`` (one fit's evidence), ``receipt`` (aggregate),
``determinism`` (compare two probe fits) and ``run`` (the full arm x seed
orchestration, resumable; ``tools/r5_critic_acceptance.sh`` drives it on the
VM). Thresholds marked ``proposed`` in :data:`PREREGISTERED` are not frozen r5
text and need the user's review before they bind a result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RECEIPT_SCHEMA = "labor-r5-critic-acceptance-receipt-v1"
EVIDENCE_SCHEMA = "labor-r5-fit-evidence-v1"
GATE_ID = "r5-critic-acceptance"
SECONDARY_GATE_ID = "G2C-all-heads"
PROTOCOL_REVISION = "G0-manifest-r5-frozen-2026-10-26T0315Z"
REPOSITORY = Path(__file__).resolve().parents[3]

FITTING_WORLDS = tuple(range(90100, 90154)) + tuple(range(90160, 90166))
CALIBRATION_WORLDS = tuple(range(90180, 90200))
RESERVED_WORLDS = tuple(range(90154, 90160))
DEVELOPMENT_WORLDS = tuple(range(2000, 2060))
SEALED_WORLDS = tuple(range(3000, 3060))
STRATA = ("all", "early_turns", "late_turns", "farmer_unlocked", "farmer_locked")
OTHER_HEADS = ("population_years", "feasibility", "settlement_guard")
SEED_RULES = ("all-seeds", "mean-with-floor")

# Every number used by a gate. "frozen" values are r5 text (manifest sections 8-10,
# intervention brief, r5-b); "proposed" values are this harness's proposals for the
# user's review and are listed in every receipt.
PREREGISTERED: dict[str, dict[str, Any]] = {
    "qol_nmae_max": {"value": 0.20, "status": "frozen", "source": "manifest s8 / r5-b 3b"},
    "qol_abs_bias_max": {"value": 0.10, "status": "frozen", "source": "manifest s8 / r5-b 3b"},
    "a_ranking_min": {"value": 0.70, "status": "frozen", "source": "manifest s8-9"},
    "a_false_improvement_max": {"value": 0.15, "status": "frozen", "source": "manifest s8-9"},
    "a_false_improvement_margin": {"value": 0.01, "status": "frozen", "source": "manifest s8-9"},
    "other_head_regression_max": {"value": 0.02, "status": "frozen", "source": "manifest s8"},
    "g2c_nmae_max": {"value": 0.20, "status": "frozen", "source": "brief s9.3 C3"},
    "g2c_abs_bias_max": {"value": 0.10, "status": "frozen", "source": "brief s9.3 C3"},
    "g2c_ranking_min": {"value": 0.70, "status": "frozen", "source": "brief s9.3 C3"},
    "vm_seeds": {"value": 5, "status": "frozen", "source": "manifest s8"},
    "local_seeds_max": {"value": 3, "status": "frozen", "source": "manifest s8"},
    "critic_epochs": {"value": 4, "status": "frozen", "source": "manifest s8 (critic3 budget)"},
    "critic_epochs_second_arm": {"value": 12, "status": "frozen", "source": "manifest s8"},
    "td_lambda": {"value": 0.8, "status": "frozen", "source": "manifest s10"},
    # Proposals (need the user's review).
    "qol_strata": {
        "value": list(STRATA),
        "status": "proposed",
        "source": "every evaluator stratum",
    },
    "td_residual_abs_bias_max": {"value": 0.05, "status": "proposed", "source": "reference scales"},
    "td_residual_mae_max": {"value": 0.20, "status": "proposed", "source": "reference scales"},
    "gap_nmae_max": {"value": 0.10, "status": "proposed", "source": "cal nMAE - fit nMAE"},
    "gap_ranking_max": {"value": 0.10, "status": "proposed", "source": "fit rank - cal rank"},
    "gap_a_ranking_max": {"value": 0.10, "status": "proposed", "source": "A held-out - reserved"},
    "seed_rule_default": {"value": "all-seeds", "status": "proposed", "source": "harness"},
    "floor_tolerance": {"value": 0.02, "status": "proposed", "source": "mean-with-floor rule"},
    "determinism_tolerance": {
        "value": 1e-6,
        "status": "proposed",
        "source": "max relative CPU drift",
    },
    "stability_thresholds": {
        "value": "labor_stability.StabilityThresholds",
        "status": "proposed",
        "source": "brief s17",
    },
    # QoL saturation monitor (r5-b item 4), relaxed for the earliest game states.
    "saturation_alarm": {
        "value": 0.05,
        "status": "approved-with-relaxation (user 26 Oct)",
        "source": "r5-b item 4; unexpected |z| > 6 share outside the early stratum",
    },
    "saturation_alarm_early": {
        "value": 0.20,
        "status": "approved-with-relaxation (user 26 Oct)",
        "source": "early stratum: first 5 decisions per (world, camp, trajectory)",
    },
    "saturation_early_decisions": {
        "value": 5,
        "status": "approved-with-relaxation (user 26 Oct)",
        "source": "early stratum size",
    },
    "saturation_expected_low_target_fraction": {
        "value": 0.05,
        "status": "approved-with-relaxation (user 26 Oct)",
        "source": "low-side saturation with G/W below this is expected (no housing/clothing)",
    },
}


def threshold(name: str) -> Any:
    return PREREGISTERED[name]["value"]


# ---------------------------------------------------------------------------
# Hashing and provenance
# ---------------------------------------------------------------------------


def file_sha256(path: Path | str | None) -> str | None:
    if path is None or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def source_record(repository: Path = REPOSITORY) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=repository, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    modules = [
        "labor_r5_acceptance.py",
        "labor_stability.py",
        "labor_bc.py",
        "labor_advantage.py",
        "labor_advantage_eval.py",
        "labor_pretrain_eval.py",
        "labor_model.py",
        "labor_training.py",
    ]
    base = Path(__file__).resolve().parent
    status = git("status", "--porcelain", "--untracked-files=no")
    bridge = os.environ.get(
        "ANTHROPOCENE_MODEL9_BRIDGE",
        str(repository / "integration" / "bin" / "model9" / "Anthropocene.Model9Bridge.dll"),
    )
    return {
        "repository": str(repository),
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
        "module_sha256": {name: file_sha256(base / name) for name in modules},
        "bridge": {"path": bridge, "sha256": file_sha256(bridge)},
        "python": sys.version.split()[0],
    }


# ---------------------------------------------------------------------------
# Check records and verdict algebra (pure, unit tested)
# ---------------------------------------------------------------------------


def check(
    check_id: str,
    group: str,
    value: Any,
    limit: Any,
    op: str,
    *,
    status: str | None = None,
    scalar: bool = True,
    every_seed: bool = False,
    detail: Any = None,
) -> dict[str, Any]:
    """One gated measurement. ``value is None`` is missing evidence: HOLD.

    ``op`` is ``<=`` or ``>=`` for scalar thresholds, ``==`` for exact
    requirements; ``status`` overrides the comparison (e.g. a nested evaluator
    verdict). ``every_seed`` marks checks the declared mean rule cannot average
    (stability, controls, bookkeeping).
    """
    if status is None:
        if value is None or limit is None:
            status = "HOLD"
        elif op == "<=":
            status = "PASS" if float(value) <= float(limit) else "FAIL"
        elif op == ">=":
            status = "PASS" if float(value) >= float(limit) else "FAIL"
        elif op == "==":
            status = "PASS" if value == limit else "FAIL"
        else:
            raise ValueError(f"unknown comparison {op!r}")
    if status not in ("PASS", "HOLD", "FAIL"):
        raise ValueError(f"unknown status {status!r}")
    return {
        "id": check_id,
        "group": group,
        "value": value,
        "threshold": limit,
        "op": op,
        "status": status,
        "scalar": bool(scalar and op in ("<=", ">=")),
        "every_seed": bool(every_seed or not scalar or op == "=="),
        "detail": detail,
    }


def combine(statuses: Sequence[str]) -> str:
    """FAIL dominates HOLD dominates PASS; no evidence at all is HOLD."""
    statuses = list(statuses)
    if not statuses:
        return "HOLD"
    if "FAIL" in statuses:
        return "FAIL"
    if "HOLD" in statuses:
        return "HOLD"
    return "PASS"


def _get(mapping: Any, *path: Any) -> Any:
    for key in path:
        if not isinstance(mapping, Mapping) or key not in mapping:
            return None
        mapping = mapping[key]
    return mapping


# --- stability ---------------------------------------------------------------------------


def stability_checks(label: str, summary: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Mandatory: an admitted stability receipt; a mandatory FAIL fails the fit."""
    if summary is None:
        return [
            check(
                f"stability:{label}",
                "stability",
                None,
                "admitted",
                "==",
                every_seed=True,
                detail="no stability receipt for this fit",
            )
        ]
    out = []
    for name, record in sorted((summary.get("checks") or {}).items()):
        if not record.get("mandatory") or record.get("status") == "NOT_APPLICABLE":
            continue
        out.append(
            check(
                f"stability:{label}:{name}",
                "stability",
                record.get("status"),
                "PASS",
                "==",
                status=record.get("status"),
                every_seed=True,
            )
        )
    injected = bool(summary.get("fault_injection"))
    out.append(
        check(
            f"stability:{label}:no-fault-injection",
            "stability",
            not injected,
            True,
            "==",
            every_seed=True,
            detail="fault-injection hooks are test-only; such a fit is never admitted",
        )
    )
    return out


# --- QoL (r5-b) and other heads --------------------------------------------------------------


def _head(
    calibration: Mapping[str, Any] | None, head: str, stratum: str
) -> Mapping[str, Any] | None:
    return _get(calibration, "calibration", "heads", head, stratum)


def qol_checks(
    calibration: Mapping[str, Any] | None,
    advantage: Mapping[str, Any] | None,
    td: Mapping[str, Any] | None,
    strata: Sequence[str] = STRATA,
) -> list[dict[str, Any]]:
    out = []
    # (a) same-village action ranking via A on the reserved worlds (primary).
    gate = _get(advantage, "gates", "ranking:qol_years")
    out.append(
        check(
            "qol:a:same-village-ranking",
            "qol",
            None if gate is None else gate.get("value"),
            threshold("a_ranking_min"),
            ">=",
            status=None if gate is None else gate.get("status"),
            detail=None
            if advantage is None
            else {
                "pairs": _get(advantage, "ranking", "qol_years", "pairs"),
                "reserved_worlds_only": advantage.get("reserved_worlds_only"),
            },
        )
    )
    if advantage is not None and not advantage.get("reserved_worlds_only"):
        out.append(
            check(
                "qol:a:reserved-worlds-only",
                "qol",
                False,
                True,
                "==",
                detail="A must be judged on the reserved worlds 90154-90159 only",
            )
        )
    # (b) bounded-head calibration per stratum.
    for stratum in strata:
        metrics = _head(calibration, "qol_years", stratum)
        nmae = None if metrics is None else metrics.get("nmae")
        bias = None if metrics is None else metrics.get("bias")
        out.append(check(f"qol:b:nmae:{stratum}", "qol", nmae, threshold("qol_nmae_max"), "<="))
        out.append(
            check(
                f"qol:b:abs-bias:{stratum}",
                "qol",
                None if bias is None else abs(float(bias)),
                threshold("qol_abs_bias_max"),
                "<=",
            )
        )
    bounded = _get(calibration, "critic", "qol_head")
    out.append(
        check(
            "qol:b:bounded-head",
            "qol",
            bounded is not None,
            True,
            "==",
            detail="QoL must use the bounded head (r5-b item 4)",
        )
    )
    # (c) TD residual consistency along each village's own trajectory.
    out.append(
        check(
            "qol:c:td-residual-abs-bias",
            "qol",
            None if td is None or td.get("bias") is None else abs(float(td["bias"])),
            threshold("td_residual_abs_bias_max"),
            "<=",
        )
    )
    out.append(
        check(
            "qol:c:td-residual-mae",
            "qol",
            None if td is None else td.get("mae"),
            threshold("td_residual_mae_max"),
            "<=",
        )
    )
    return out


def population_ranking_checks(
    calibration: Mapping[str, Any] | None, strata: Sequence[str] = STRATA
) -> list[dict[str, Any]]:
    """r5-b item 3: cross-village ranking remains a criterion for population-years only."""
    out = []
    for stratum in strata:
        metrics = _head(calibration, "population_years", stratum)
        out.append(
            check(
                f"population_years:ranking:{stratum}",
                "population-years",
                _get(metrics, "ranking", "accuracy"),
                threshold("g2c_ranking_min"),
                ">=",
            )
        )
    return out


def regression_checks(
    calibration: Mapping[str, Any] | None,
    control: Mapping[str, Any] | None,
    strata: Sequence[str] = STRATA,
) -> list[dict[str, Any]]:
    """Other heads: no regression beyond 0.02 against the critic3 control."""
    limit = threshold("other_head_regression_max")
    out = []
    for head in OTHER_HEADS:
        for stratum in strata:
            new = _head(calibration, head, stratum)
            old = _get(control, "calibration", "heads", head, stratum)
            if new is None and old is None:
                continue
            base = f"regression:{head}:{stratum}"
            if new is None or old is None:
                out.append(
                    check(
                        base,
                        "other-heads",
                        None,
                        limit,
                        "<=",
                        detail="stratum missing in the candidate or the control",
                    )
                )
                continue
            if new.get("zero_target_head") or old.get("zero_target_head"):
                out.append(
                    check(
                        f"{base}:mean-abs-pred",
                        "other-heads",
                        _delta(new.get("mean_abs_pred"), old.get("mean_abs_pred")),
                        limit,
                        "<=",
                    )
                )
                continue
            out.append(
                check(
                    f"{base}:nmae",
                    "other-heads",
                    _delta(new.get("nmae"), old.get("nmae")),
                    limit,
                    "<=",
                )
            )
            out.append(
                check(
                    f"{base}:abs-bias",
                    "other-heads",
                    _delta(_abs(new.get("bias")), _abs(old.get("bias"))),
                    limit,
                    "<=",
                )
            )
            rank_new = _get(new, "ranking", "accuracy")
            rank_old = _get(old, "ranking", "accuracy")
            if rank_new is not None or rank_old is not None:
                out.append(
                    check(
                        f"{base}:ranking",
                        "other-heads",
                        _delta(rank_old, rank_new),
                        limit,
                        "<=",
                        detail="control ranking minus candidate ranking",
                    )
                )
    return out


def _delta(a: Any, b: Any) -> float | None:
    return None if a is None or b is None else float(a) - float(b)


def _abs(a: Any) -> float | None:
    return None if a is None else abs(float(a))


def g2c_checks(
    calibration: Mapping[str, Any] | None, strata: Sequence[str] = STRATA
) -> list[dict[str, Any]]:
    """Full G2C per head and stratum (the evaluator's own gate), the secondary gate."""
    out = []
    for head in ("qol_years", *OTHER_HEADS):
        for stratum in strata:
            metrics = _head(calibration, head, stratum)
            out.append(
                check(
                    f"g2c:{head}:{stratum}",
                    "g2c",
                    None if metrics is None else metrics.get("gate"),
                    "PASS",
                    "==",
                    status=None if metrics is None else metrics.get("gate"),
                    detail=None
                    if metrics is None
                    else {k: metrics.get(k) for k in ("nmae", "bias", "worlds", "rows")}
                    | {"ranking": _get(metrics, "ranking", "accuracy")},
                )
            )
    return out


# --- advantage head --------------------------------------------------------------------------


def advantage_checks(advantage: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The evaluator's required gates, one check each (reserved worlds)."""
    if advantage is None:
        return [check("a:report", "advantage", None, "PASS", "==", detail="no A evaluation report")]
    out = []
    scalar = {"false_improvement": "<=", "idle_control": ">=", "delayed_benefit": ">="}
    for key in advantage.get("required_gates", []):
        gate = advantage["gates"].get(key) or {}
        op = scalar.get(key, ">=" if key.startswith("ranking:") else "<=")
        controls = key in ("idle_control", "delayed_benefit", "bookkeeping_invariance")
        out.append(
            check(
                f"a:{key}",
                "advantage",
                gate.get("value"),
                gate.get("threshold"),
                op,
                status=gate.get("status"),
                scalar=not controls,
                every_seed=controls,
            )
        )
    out.append(
        check(
            "a:reserved-worlds-only",
            "advantage",
            bool(advantage.get("reserved_worlds_only")),
            True,
            "==",
        )
    )
    return out


# --- generalisation gap ------------------------------------------------------------------------


def gap_checks(
    calibration: Mapping[str, Any] | None,
    advantage_fit: Mapping[str, Any] | None,
    advantage: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    out = []
    gaps = _get(calibration, "generalisation_gap")
    for head in ("qol_years", *OTHER_HEADS):
        record = None if gaps is None else gaps.get(head)
        nmae_gap = None if record is None else record.get("nmae_cal_minus_fit")
        out.append(check(f"gap:{head}:nmae", "gap", nmae_gap, threshold("gap_nmae_max"), "<="))
        rank_gap = None if record is None else record.get("ranking_cal_minus_fit")
        if record is None or rank_gap is not None:
            out.append(
                check(
                    f"gap:{head}:ranking",
                    "gap",
                    None if rank_gap is None else -float(rank_gap),
                    threshold("gap_ranking_max"),
                    "<=",
                    detail="fit ranking minus calibration ranking",
                )
            )
    held = _get(advantage_fit, "heldout_ranking", "qol_years")
    reserved = _get(advantage, "gates", "ranking:qol_years", "value")
    out.append(
        check(
            "gap:advantage:qol_years",
            "gap",
            None if held is None or reserved is None else float(held) - float(reserved),
            threshold("gap_a_ranking_max"),
            "<=",
            detail="A held-out (fitting-world split) ranking minus reserved-world ranking",
        )
    )
    return out


def fit_checks(evidence: Mapping[str, Any], *, strata: Sequence[str] = STRATA) -> dict[str, Any]:
    """Every check of one (arm, seed) fit, grouped into the primary and secondary gates."""
    primary = []
    primary += stability_checks("critic", _get(evidence, "stability", "critic"))
    primary += stability_checks("advantage", _get(evidence, "stability", "advantage"))
    primary += qol_checks(
        evidence.get("calibration"),
        evidence.get("advantage_eval"),
        evidence.get("td_residual"),
        strata,
    )
    primary += population_ranking_checks(evidence.get("calibration"), strata)
    primary += regression_checks(evidence.get("calibration"), evidence.get("control"), strata)
    if evidence.get("control") is None:
        primary.append(
            check(
                "regression:control",
                "other-heads",
                None,
                "critic3",
                "==",
                detail="no critic3 control report: other-head regression is unproven",
            )
        )
    primary += advantage_checks(evidence.get("advantage_eval"))
    primary += gap_checks(
        evidence.get("calibration"), evidence.get("advantage_fit"), evidence.get("advantage_eval")
    )
    secondary = g2c_checks(evidence.get("calibration"), strata)
    return {
        "primary": primary,
        "secondary": secondary,
        "primary_status": combine([c["status"] for c in primary]),
        "secondary_status": combine([c["status"] for c in secondary]),
    }


# --- seed rule -------------------------------------------------------------------------------


def seed_verdict(
    per_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    rule: str,
    required_seeds: int,
    floor_tolerance: float | None = None,
) -> dict[str, Any]:
    """Aggregate one gate's per-seed checks under the declared seed rule.

    ``all-seeds``: PASS only when every seed passes every check. ``mean-with-floor``:
    scalar checks pass on the seed mean with the worst seed within
    ``floor_tolerance`` of the threshold; non-scalar and every-seed checks (stability,
    controls, bookkeeping, exact requirements) must pass in every seed. Missing
    seeds (fewer than ``required_seeds``) HOLD under both rules.
    """
    if rule not in SEED_RULES:
        raise ValueError(f"seed rule must be one of {SEED_RULES}")
    tolerance = threshold("floor_tolerance") if floor_tolerance is None else float(floor_tolerance)
    seeds = sorted(per_seed)
    by_id: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for seed in seeds:
        for record in per_seed[seed]:
            by_id[record["id"]][seed] = record
    rows = []
    for check_id, records in sorted(by_id.items()):
        statuses = {seed: records[seed]["status"] if seed in records else "HOLD" for seed in seeds}
        example = next(iter(records.values()))
        values = [
            float(r["value"])
            for r in records.values()
            if r["scalar"] and r["value"] is not None and _finite(r["value"])
        ]
        spread = (
            {
                "mean": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
                "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }
            if values
            else None
        )
        every = combine(list(statuses.values()))
        if rule == "all-seeds" or example["every_seed"] or not example["scalar"]:
            status = every
            basis = "every-seed"
        else:
            missing = len(values) != len(seeds) or any(r["value"] is None for r in records.values())
            limit = float(example["threshold"])
            if missing:
                status = "HOLD"
            elif example["op"] == "<=":
                worst = max(values)
                status = (
                    "PASS" if spread["mean"] <= limit and worst <= limit + tolerance else "FAIL"
                )
            else:
                worst = min(values)
                status = (
                    "PASS" if spread["mean"] >= limit and worst >= limit - tolerance else "FAIL"
                )
            basis = "mean-with-floor"
        rows.append(
            {
                "id": check_id,
                "group": example["group"],
                "op": example["op"],
                "threshold": example["threshold"],
                "per_seed": {
                    str(s): {
                        "value": records[s]["value"] if s in records else None,
                        "status": statuses[s],
                    }
                    for s in seeds
                },
                "spread": spread,
                "passes_every_seed": every == "PASS",
                "status": status,
                "basis": basis,
            }
        )
    statuses = [row["status"] for row in rows]
    coverage = "PASS" if len(seeds) >= required_seeds else "HOLD"
    per_seed_status = {str(s): combine([r["status"] for r in per_seed[s]]) for s in seeds}
    return {
        "rule": rule,
        "floor_tolerance": tolerance if rule == "mean-with-floor" else None,
        "seeds": seeds,
        "required_seeds": int(required_seeds),
        "seed_coverage": coverage,
        "per_seed_status": per_seed_status,
        "passes_in_every_seed": all(v == "PASS" for v in per_seed_status.values()) and bool(seeds),
        "checks": rows,
        "status": combine(statuses + [coverage]),
    }


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


# --- arms -----------------------------------------------------------------------------------


COMPARABILITY_KEYS = (
    "dataset_digests",
    "value_support",
    "return_tail",
    "epochs",
    "batch_size",
    "lr",
    "max_grad_norm",
    "qol_head",
    "advantage_config",
    "payload_digests",
)


def arm_comparison(
    candidate: Mapping[int, Mapping[str, Any]], control: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    """Paired per-seed effect of privileged inputs and the same-budget comparability check."""
    seeds = sorted(set(candidate) & set(control))
    mismatches = []
    for seed in seeds:
        a, b = candidate[seed].get("comparability", {}), control[seed].get("comparability", {})
        for key in COMPARABILITY_KEYS:
            if a.get(key) != b.get(key):
                mismatches.append(
                    {"seed": seed, "field": key, "candidate": a.get(key), "control": b.get(key)}
                )
        if not candidate[seed].get("privileged") or control[seed].get("privileged"):
            mismatches.append(
                {
                    "seed": seed,
                    "field": "privileged",
                    "candidate": candidate[seed].get("privileged"),
                    "control": control[seed].get("privileged"),
                }
            )
    metrics = {
        "qol:b:nmae:all": ("calibration", "calibration", "heads", "qol_years", "all", "nmae"),
        "qol:b:bias:all": ("calibration", "calibration", "heads", "qol_years", "all", "bias"),
        "population_years:nmae:all": (
            "calibration",
            "calibration",
            "heads",
            "population_years",
            "all",
            "nmae",
        ),
        "population_years:ranking:all": (
            "calibration",
            "calibration",
            "heads",
            "population_years",
            "all",
            "ranking",
            "accuracy",
        ),
        "qol:a:ranking": ("advantage_eval", "gates", "ranking:qol_years", "value"),
        "a:false_improvement": ("advantage_eval", "gates", "false_improvement", "value"),
        "qol:c:td-residual-mae": ("td_residual", "mae"),
    }
    effects = {}
    for name, path in metrics.items():
        diffs = []
        for seed in seeds:
            a, b = _get(candidate[seed], *path), _get(control[seed], *path)
            if a is not None and b is not None:
                diffs.append(float(a) - float(b))
        effects[name] = (
            None
            if not diffs
            else {
                "paired_seeds": len(diffs),
                "mean_privileged_minus_control": statistics.fmean(diffs),
                "sd": statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
                "per_seed": diffs,
                "sign_consistent": all(d > 0 for d in diffs) or all(d < 0 for d in diffs),
            }
        )
    status = "HOLD" if not seeds or mismatches else "PASS"
    return {
        "paired_seeds": seeds,
        "comparability": {
            "status": status,
            "mismatches": mismatches,
            "fields": list(COMPARABILITY_KEYS),
        },
        "effects": effects,
        "status": status,
    }


# --- determinism ---------------------------------------------------------------------------


def compare_state_dicts(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    keys = sorted(set(a) | set(b))
    if set(a) != set(b):
        return {"bitwise": False, "max_relative": None, "missing": sorted(set(a) ^ set(b))}
    bitwise, worst = True, 0.0
    for key in keys:
        x, y = a[key], b[key]
        if not isinstance(x, torch.Tensor):
            if x != y:
                bitwise = False
            continue
        if not torch.equal(x, y):
            bitwise = False
            if x.is_floating_point():
                diff = (x.double() - y.double()).abs().max()
                scale = x.double().abs().max().clamp_min(1e-12)
                worst = max(worst, float(diff / scale))
            else:
                worst = math.inf
    return {"bitwise": bitwise, "max_relative": worst, "tensors": len(keys)}


def determinism_checks(report: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if report is None:
        return [
            check(
                "determinism", "determinism", None, "bitwise", "==", detail="no determinism probe"
            )
        ]
    out = []
    tolerance = threshold("determinism_tolerance")
    for name, result in sorted(report.get("comparisons", {}).items()):
        if result.get("bitwise"):
            out.append(
                check(
                    f"determinism:{name}", "determinism", "bitwise", "bitwise", "==", detail=result
                )
            )
        else:
            value = result.get("max_relative")
            out.append(
                check(
                    f"determinism:{name}",
                    "determinism",
                    value,
                    tolerance,
                    "<=",
                    every_seed=True,
                    detail={"note": "within the documented CPU tolerance, not bitwise", **result},
                )
            )
    return out or [check("determinism", "determinism", None, "bitwise", "==")]


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmSpec:
    name: str
    privileged: bool
    epochs: int
    role: str  # "candidate" or "control"

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "privileged": self.privileged,
            "epochs": self.epochs,
            "role": self.role,
        }


DEFAULT_ARMS = (
    ArmSpec("privileged-e4", True, 4, "candidate"),
    ArmSpec("control-e4", False, 4, "control"),
)


def parse_arms(text: str | None) -> tuple[ArmSpec, ...]:
    """``name:privileged|plain:epochs:candidate|control`` comma list (default: the r5 pair)."""
    if not text:
        return DEFAULT_ARMS
    arms = []
    for part in text.split(","):
        name, kind, epochs, role = part.split(":")
        if kind not in ("privileged", "plain") or role not in ("candidate", "control"):
            raise ValueError(f"bad arm {part!r}")
        arms.append(ArmSpec(name, kind == "privileged", int(epochs), role))
    if sum(a.role == "candidate" for a in arms) < 1:
        raise ValueError("at least one candidate arm is required")
    return tuple(arms)


def build_receipt(
    evidence: Sequence[Mapping[str, Any]],
    *,
    arms: Sequence[ArmSpec],
    seeds: Sequence[int],
    seed_rule: str,
    required_seeds: int,
    determinism: Mapping[str, Any] | None,
    configuration: Mapping[str, Any],
    attempt_id: str,
    source: Mapping[str, Any] | None = None,
    strata: Sequence[str] = STRATA,
    floor_tolerance: float | None = None,
) -> dict[str, Any]:
    """The gate receipt (brief section 4 fields) with the machine verdict."""
    by_arm: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for record in evidence:
        by_arm[record["arm"]][int(record["seed"])] = record
    arm_reports = {}
    for arm in arms:
        fits = by_arm.get(arm.name, {})
        per_seed_primary, per_seed_secondary = {}, {}
        for seed in seeds:
            if seed in fits:
                checks = fit_checks(fits[seed], strata=strata)
                per_seed_primary[seed] = checks["primary"]
                per_seed_secondary[seed] = checks["secondary"]
            else:
                missing = [check("fit:missing", "fit", None, "present", "==", every_seed=True)]
                per_seed_primary[seed] = missing
                per_seed_secondary[seed] = missing
        arm_reports[arm.name] = {
            "arm": arm.state_dict(),
            GATE_ID: seed_verdict(
                per_seed_primary,
                rule=seed_rule,
                required_seeds=required_seeds,
                floor_tolerance=floor_tolerance,
            ),
            SECONDARY_GATE_ID: seed_verdict(
                per_seed_secondary,
                rule=seed_rule,
                required_seeds=required_seeds,
                floor_tolerance=floor_tolerance,
            ),
        }
    candidates = [a for a in arms if a.role == "candidate"]
    controls = [a for a in arms if a.role == "control"]
    comparisons = {}
    for cand in candidates:
        for ctrl in controls:
            if cand.epochs != ctrl.epochs:
                continue
            comparisons[f"{cand.name}-vs-{ctrl.name}"] = arm_comparison(
                by_arm.get(cand.name, {}), by_arm.get(ctrl.name, {})
            )
    determinism_records = determinism_checks(determinism)
    determinism_status = combine([c["status"] for c in determinism_records])
    primary_arm = candidates[0]
    gate_status = arm_reports[primary_arm.name][GATE_ID]["status"]
    comparison_status = combine(
        [
            c["status"]
            for name, c in comparisons.items()
            if name.startswith(primary_arm.name + "-vs-")
        ]
    )
    exclusions = _historical_exclusions(evidence)
    smoke = bool(configuration.get("smoke"))
    verdict = combine(
        [gate_status, determinism_status, comparison_status, exclusions["status"]]
        + (["HOLD"] if smoke else [])
    )
    unresolved = []
    if smoke:
        unresolved.append(
            {
                "check": "configuration:smoke",
                "status": "HOLD",
                "detail": "smoke run without the randomized tail is not admissible evidence",
            }
        )
    if exclusions["status"] != "PASS":
        unresolved.append(
            {
                "check": "partitions:historical-fitting-exclusions",
                "status": "FAIL",
                "violations": exclusions["violations"],
            }
        )
    for arm_name, report in arm_reports.items():
        for row in report[GATE_ID]["checks"]:
            if row["status"] != "PASS" and arm_name == primary_arm.name:
                unresolved.append({"arm": arm_name, "check": row["id"], "status": row["status"]})
    for name, comparison in comparisons.items():
        if comparison["status"] != "PASS":
            unresolved.append(
                {
                    "comparison": name,
                    "status": comparison["status"],
                    "mismatches": comparison["comparability"]["mismatches"][:10],
                }
            )
    for row in determinism_records:
        if row["status"] != "PASS":
            unresolved.append({"check": row["id"], "status": row["status"]})
    artifacts = {f"{r['arm']}/seed-{r['seed']}": r.get("artifacts", {}) for r in evidence}
    bindings = {f"{r['arm']}/seed-{r['seed']}": r.get("bindings", {}) for r in evidence}
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "gate_id": GATE_ID,
        "secondary_gate_id": SECONDARY_GATE_ID,
        "attempt_id": attempt_id,
        "protocol_revision": PROTOCOL_REVISION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": source if source is not None else source_record(),
        "configuration": {**configuration, "configuration_digest": json_digest(configuration)},
        "artifacts": artifacts,
        "bindings": bindings,
        "partitions": {
            "fitting": [min(FITTING_WORLDS), 90153, 90160, max(FITTING_WORLDS)],
            "fitting_ranges": "90100-90153,90160-90165",
            "calibration": "90180-90199",
            "reserved_heldout_tests_only": "90154-90159",
            "development": "2000-2059",
            "sealed": "3000-3059",
            "historical_fitting_exclusions": exclusions,
        },
        "preregistered": PREREGISTERED,
        "proposed_for_review": sorted(
            k for k, v in PREREGISTERED.items() if v["status"] == "proposed"
        ),
        "seed_rule": {
            "rule": seed_rule,
            "declared_alternative": [r for r in SEED_RULES if r != seed_rule],
            "required_seeds": int(required_seeds),
        },
        "arms": arm_reports,
        "arm_comparisons": comparisons,
        "determinism": {
            "report": determinism,
            "checks": determinism_records,
            "status": determinism_status,
        },
        "work": _work_totals(evidence),
        "machine_state": verdict,
        "machine_verdict": verdict,
        "verdict_basis": {
            "candidate_arm": primary_arm.name,
            GATE_ID: gate_status,
            "arm_comparison": comparison_status,
            "determinism": determinism_status,
            "historical_fitting_exclusions": exclusions["status"],
            "smoke": smoke,
            SECONDARY_GATE_ID: arm_reports[primary_arm.name][SECONDARY_GATE_ID]["status"],
            "secondary_gate_note": "full G2C for every head is reported; the r5 acceptance gate "
            "judges QoL per r5-b and the other heads by no-regression (intervention brief)",
        },
        "reviewers": {
            "mathematical_statistical": {
                "identity": None,
                "model": None,
                "effort": None,
                "verdict": "NOT_RUN",
            },
            "runtime_provenance": {
                "identity": None,
                "model": None,
                "effort": None,
                "verdict": "NOT_RUN",
            },
        },
        "unresolved_findings": unresolved,
        "next_authorized_transition": (
            "admission step 3: fit and calibrate Q under the frozen actor (G5-style); no actor "
            "update through Q (manifest s8 admission order) -- pending two reviewer PASS verdicts"
            if verdict == "PASS"
            else "none: the gate is closed; a corrected attempt needs a new manifest/attempt id"
        ),
    }
    receipt["receipt_digest"] = json_digest(receipt)
    return receipt


def _historical_exclusions(evidence: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Calibration, reserved, development and sealed worlds must not appear in any fit."""
    banned = (
        set(CALIBRATION_WORLDS)
        | set(RESERVED_WORLDS)
        | set(DEVELOPMENT_WORLDS)
        | set(SEALED_WORLDS)
    )
    violations = {}
    for record in evidence:
        fitted = set(
            int(w) for w in (_get(record, "bindings", "critic_training_world_seeds") or [])
        )
        fitted |= set(int(w) for w in (_get(record, "advantage_fit", "fit_worlds") or []))
        fitted |= set(int(w) for w in (_get(record, "advantage_fit", "heldout_worlds") or []))
        bad = sorted(fitted & banned)
        if bad:
            violations[f"{record['arm']}/seed-{record['seed']}"] = bad
    return {
        "checked_fits": len(evidence),
        "violations": violations,
        "status": "FAIL" if violations else "PASS",
    }


def _work_totals(evidence: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seconds = defaultdict(float)
    steps = defaultdict(int)
    for record in evidence:
        for stage, value in (record.get("timings_seconds") or {}).items():
            seconds[stage] += float(value)
        for label in ("critic", "advantage"):
            summary = _get(record, "stability", label)
            if summary:
                steps[label] += int(summary.get("steps", 0))
    return {
        "seconds_by_stage": dict(seconds),
        "optimizer_steps": dict(steps),
        "fits": len(evidence),
    }


# ---------------------------------------------------------------------------
# Evaluation of one fit
# ---------------------------------------------------------------------------


def td_residual_report(
    task: Any, costs: Any, dataset: Mapping[str, Any], batch_size: int = 256
) -> dict[str, Any]:
    """TD residual consistency of the QoL head along each village's own trajectory.

    delta_i = r_i + gamma_i V(s_{i+1}) - V(s_i) in QoL reference scales, where the
    state after a trajectory's last row is valued by its declared tail (zero at a
    true terminal) and trajectories are keyed exactly as ``discounted_returns``.
    Means are world-balanced over admitted rows, like nMAE/bias.
    """
    import numpy as np

    from .labor_bc import (
        CRITIC_DISCOUNT_YEARS,
        RETURN_TAIL_RANDOMIZED,
        TAIL_METHOD_FIELD,
        TAIL_RANDOMIZED,
        TAIL_TERMINAL_EXACT,
        TAIL_VALUE_PREFIX,
        TRAJECTORY_FIELD,
        critic_target_available,
        elapsed_years,
        return_tail_method,
    )
    from .labor_model import LABOR_OBJECTIVE_REFERENCE_SCALES
    from .labor_pretrain_eval import _np, critic_predictions, world_balanced

    n = int(dataset["row_world"].shape[0])
    every = np.arange(n)
    values = critic_predictions(task, costs, dataset, every, batch_size)["qol_years"]
    reward = _np(dataset, "reward_qol_years").reshape(-1).astype(np.float64)
    elapsed = elapsed_years(dataset).reshape(-1).double().numpy()
    terminal = _np(dataset, "transition_terminal").reshape(-1).astype(bool)
    turns = _np(dataset, "row_turn").reshape(-1)
    worlds = _np(dataset, "row_world").reshape(-1)
    camps = _np(dataset, "row_camp").reshape(-1)
    trajectories = (
        _np(dataset, TRAJECTORY_FIELD).reshape(-1)
        if TRAJECTORY_FIELD in dataset
        else np.full(n, -1)
    )
    randomized = return_tail_method(dataset) == RETURN_TAIL_RANDOMIZED
    codes = _np(dataset, TAIL_METHOD_FIELD).reshape(-1) if randomized else None
    tails = _np(dataset, TAIL_VALUE_PREFIX + "qol_years").reshape(-1) if randomized else None
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i in range(n):
        groups[(int(worlds[i]), int(camps[i]), int(trajectories[i]))].append(i)
    delta = np.zeros(n)
    for indices in groups.values():
        indices.sort(key=lambda i: turns[i])
        for position, i in enumerate(indices):
            gamma = math.exp(-float(elapsed[i]) / CRITIC_DISCOUNT_YEARS)
            if terminal[i]:
                following = 0.0
            elif position + 1 < len(indices):
                following = float(values[indices[position + 1]])
            elif randomized and codes[i] in (TAIL_TERMINAL_EXACT, TAIL_RANDOMIZED):
                following = float(tails[i])
            else:
                following = 0.0
            delta[i] = reward[i] + gamma * following - float(values[i])
    scale = LABOR_OBJECTIVE_REFERENCE_SCALES["qol_years"]
    keep = ~_np(dataset, "row_quarantined").reshape(-1).astype(bool) & critic_target_available(
        dataset
    ).numpy().astype(bool)
    rows = np.flatnonzero(keep)
    weights = world_balanced(worlds[rows], np.ones(len(rows), dtype=bool))
    normalized = delta[rows] / scale
    total = float(weights.sum())
    early = None
    if len(rows):
        from .labor_pretrain_eval import decision_index

        index, count = decision_index(worlds[rows], turns[rows])
        early = index < (count + 1) // 2

    def mean(values_: np.ndarray, mask: np.ndarray | None = None) -> float | None:
        w = weights if mask is None else weights * mask
        s = float(w.sum())
        return None if s <= 0 else float((values_ * w).sum() / s)

    return {
        "rows": int(len(rows)),
        "worlds": int(len(np.unique(worlds[rows]))) if len(rows) else 0,
        "scale": scale,
        "units": "QoL reference scales",
        "bias": mean(normalized) if total > 0 else None,
        "mae": mean(np.abs(normalized)) if total > 0 else None,
        "strata": {
            "early_turns": {"bias": mean(normalized, early), "mae": mean(np.abs(normalized), early)}
            if early is not None
            else None,
            "late_turns": {
                "bias": mean(normalized, ~early),
                "mae": mean(np.abs(normalized), ~early),
            }
            if early is not None
            else None,
        },
        "trajectory_keys": "world, camp, row_trajectory (as labor_bc.discounted_returns)",
    }


def _stability_from_checkpoint(state: Mapping[str, Any]) -> Mapping[str, Any] | None:
    return _get(state, "heuristic_critic_pretraining", "stability")


def evaluate_fit(
    *,
    arm: str,
    seed: int,
    checkpoint: Path,
    advantage: Path,
    calibration_data: Path,
    fit_data: Path | None,
    reserved_payloads: Sequence[Path],
    control_report: Mapping[str, Any] | None = None,
    privileged: bool | None = None,
    timings: Mapping[str, float] | None = None,
    batch_size: int = 256,
) -> dict[str, Any]:
    """One (arm, seed) fit's evidence: calibration, TD residual, A on reserved worlds, stability."""
    import torch

    from .labor_advantage import load_advantage_head, load_candidate_shards
    from .labor_advantage_eval import evaluate_advantage
    from .labor_pretrain_eval import _load_dataset, _load_models, run_critic_calibration
    from .labor_training import _normalization_digest
    from .labor_value_calibration import labor_critic_digest

    started = time.time()
    state, actor, task, costs, normalizer = _load_models(checkpoint)
    calibration = run_critic_calibration(checkpoint, calibration_data, fit_data, batch_size)
    calibration["critic"] = {
        "qol_head": None if task.qol_head is None else task.qol_head.state_dict(),
        "value_support": task.support.state_dict(),
        "privileged": task.privileged_layout is not None,
    }
    dataset = _load_dataset(
        calibration_data, normalizer, require_privileged=task.privileged_layout is not None
    )
    td = td_residual_report(task, costs, dataset, batch_size)
    adv_state = torch.load(advantage, map_location="cpu", weights_only=True)
    for module in (task, costs):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    head, adv_config = load_advantage_head(adv_state, task, costs)
    reserved = load_candidate_shards(list(reserved_payloads))
    adv_fit = adv_state.get("fit", {})
    adv_eval = evaluate_advantage(
        task,
        costs,
        head,
        reserved,
        horizons=adv_config.horizons,
        fit_worlds=adv_fit.get("fit_worlds", []),
        se_multiplier=adv_config.se_multiplier,
    )
    critic_summary = state.get("heuristic_critic_pretraining", {})
    sidecar = settings_path(checkpoint)
    fit_settings = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    comparability = {
        "dataset_digests": critic_summary.get("dataset_digest"),
        "value_support": task.support.state_dict(),
        "return_tail": critic_summary.get("return_tail"),
        "return_tail_settings": critic_summary.get("return_tail_settings"),
        "epochs": critic_summary.get("epochs"),
        "batch_size": fit_settings.get("batch_size"),
        "lr": fit_settings.get("lr"),
        "max_grad_norm": fit_settings.get("max_grad_norm"),
        "qol_head": None if task.qol_head is None else task.qol_head.state_dict(),
        "advantage_config": adv_config.state_dict() | {"seed": None},
        "payload_digests": sorted(adv_fit.get("payload_digests", []) or []),
    }
    timings = dict(timings or {})
    timings["evaluate"] = time.time() - started
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "arm": arm,
        "seed": int(seed),
        "privileged": task.privileged_layout is not None
        if privileged is None
        else bool(privileged),
        "artifacts": {
            "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint)},
            "advantage": {"path": str(advantage), "sha256": file_sha256(advantage)},
            "calibration_dataset": {
                "path": str(calibration_data),
                "digest": dataset["collection"]["dataset_digest"],
            },
            "fit_dataset": None if fit_data is None else str(fit_data),
            "reserved_payloads": [
                {"path": str(p), "sha256": file_sha256(p) if Path(p).is_file() else None}
                for p in reserved_payloads
            ],
        },
        "bindings": {
            "critic_digest": labor_critic_digest(task, costs),
            "normalizer_digest": _normalization_digest(normalizer),
            "privileged_normalizer_digest": None
            if task.privileged_layout is None
            else task.privileged_normalizer.digest(),
            "actor_parameter_sha256": _module_sha(actor),
            "controller": (
                "H12 teacher (rho) continuations; randomized tail prefix 25 / 16 draws / "
                "rho 0.95 / cap 200"
            ),
            "critic_training_world_seeds": list(state.get("critic_training_world_seeds", [])),
            "advantage_critic_digest": adv_state.get("critic_digest"),
        },
        "comparability": comparability,
        "calibration": calibration,
        "td_residual": td,
        "advantage_fit": {
            k: v for k, v in adv_fit.items() if k not in ("stability", "loss_history")
        },
        "advantage_eval": adv_eval,
        "control": control_report,
        "stability": {
            "critic": _stability_from_checkpoint(state),
            "advantage": adv_fit.get("stability"),
        },
        "timings_seconds": timings,
    }
    return evidence


def _module_sha(module: Any) -> str:
    from .labor_stability import module_digest

    return module_digest(module)


# ---------------------------------------------------------------------------
# Orchestration (arm x seed), resumable
# ---------------------------------------------------------------------------


def _python() -> list[str]:
    return [sys.executable, "-m"]


def _run(command: Sequence[str], log: Path, threads: int) -> None:
    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS=str(threads),
        MKL_NUM_THREADS=str(threads),
        OPENBLAS_NUM_THREADS=str(threads),
        PYTHONUNBUFFERED="1",
    )
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPOSITORY / "src"), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(shlex.quote(str(c)) for c in command)}\n")
        handle.flush()
        result = subprocess.run(
            list(map(str, command)), stdout=handle, stderr=subprocess.STDOUT, env=env
        )
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def settings_path(checkpoint: Path) -> Path:
    return Path(checkpoint).with_suffix(".settings.json")


def _stamp_fit_settings(path: Path, settings: Mapping[str, Any]) -> None:
    """Record the exact fit schedule beside the critic checkpoint (comparability evidence)."""
    record = {**settings, "checkpoint_sha256": file_sha256(path)}
    settings_path(path).write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")


def _tail_flag(args: argparse.Namespace) -> list[str]:
    """r5 freeze: every critic target uses the randomized tail; only --smoke relaxes it (HOLD)."""
    return [] if getattr(args, "smoke", False) else ["--require-randomized-tail"]


def run_fit(args: argparse.Namespace, arm: ArmSpec, seed: int) -> Path:
    """initialize -> pretrain-critic (exact resume) -> A fit -> evaluate; done steps are skipped."""
    work = Path(args.out) / arm.name / f"seed-{seed}"
    work.mkdir(parents=True, exist_ok=True)
    log = work / "fit.log"
    timings_path = work / "timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}
    threads = int(args.threads)

    def timed(stage: str, function) -> None:
        start = time.time()
        function()
        timings[stage] = timings.get(stage, 0.0) + time.time() - start
        timings_path.write_text(json.dumps(timings, indent=1))

    init = work / "checkpoint-000.pt"
    if not init.exists():
        command = [
            *_python(),
            "anthropocene_rl.autonomous_v7.labor_training",
            "initialize",
            "--campaign-r5",
            "--normalizer-checkpoint",
            args.normalizer_checkpoint,
            "--world-seed",
            str(args.init_world),
            "--optimizer-seed",
            str(seed),
            "--qol-head",
            args.qol_head,
            "--out",
            init,
        ]
        if arm.privileged:
            command += ["--privileged-normalizer", args.privileged_normalizer]
        command += shlex.split(args.init_args or "")
        timed("initialize", lambda: _run(command, log, threads))
    critic = work / "critic.pt"
    if not critic.exists():
        command = [
            *_python(),
            "anthropocene_rl.autonomous_v7.labor_training",
            "pretrain-critic",
            "--checkpoint",
            init,
            "--epochs",
            str(arm.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--max-grad-norm",
            str(args.max_grad_norm),
            *_tail_flag(args),
            "--stability-log",
            work / "stability-critic.jsonl",
            "--stability-every",
            str(args.stability_every),
            "--resume-state",
            work / "critic.resume.pt",
            "--resume-every",
            str(args.resume_every),
            "--progress",
            "--out",
            critic,
        ]
        for directory in args.fit_data:
            command += ["--dataset", directory]
        if args.learner_state_continuations:
            command += ["--learner-state-continuations"]
        command += shlex.split(args.critic_args or "")

        def fit_critic() -> None:
            _run(command, log, threads)
            _stamp_fit_settings(
                critic,
                {
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "max_grad_norm": args.max_grad_norm,
                    "epochs": arm.epochs,
                    "arm": arm.name,
                    "seed": seed,
                },
            )

        timed("pretrain_critic", fit_critic)
    advantage = work / "advantage.pt"
    if not advantage.exists():
        command = [
            *_python(),
            "anthropocene_rl.autonomous_v7.labor_advantage",
            "fit",
            "--checkpoint",
            critic,
            "--payloads",
            *args.fit_payloads,
            "--out",
            advantage,
            "--epochs",
            str(args.advantage_epochs),
            "--seed",
            str(seed),
            "--stability-log",
            work / "stability-advantage.jsonl",
            "--stability-every",
            str(args.stability_every),
        ]
        command += shlex.split(args.advantage_args or "")
        timed("advantage_fit", lambda: _run(command, log, threads))
    evidence_path = work / "evidence.json"
    if not evidence_path.exists():
        command = [
            *_python(),
            "anthropocene_rl.autonomous_v7.labor_r5_acceptance",
            "evaluate",
            "--arm",
            arm.name,
            "--seed",
            str(seed),
            "--checkpoint",
            critic,
            "--advantage",
            advantage,
            "--calibration-data",
            args.calibration_data,
            "--fit-data",
            args.fit_data[0],
            "--reserved-payloads",
            *args.reserved_payloads,
            "--timings",
            timings_path,
            "--out",
            evidence_path,
        ]
        if args.control_report:
            command += ["--control-report", args.control_report]
        timed("evaluate", lambda: _run(command, log, threads))
    return evidence_path


def run_determinism(args: argparse.Namespace, arm: ArmSpec, seed: int) -> Path:
    """Two identical short critic fits and two identical A fits from the same inputs."""
    import torch

    work = Path(args.out) / "determinism"
    work.mkdir(parents=True, exist_ok=True)
    out = work / "determinism.json"
    if out.exists():
        return out
    init = Path(args.out) / arm.name / f"seed-{seed}" / "checkpoint-000.pt"
    critic_full = Path(args.out) / arm.name / f"seed-{seed}" / "critic.pt"
    log = work / "determinism.log"
    comparisons = {}
    critic_outputs = []
    for repeat in (0, 1):
        target = work / f"critic-probe-{repeat}.pt"
        if not target.exists():
            command = [
                *_python(),
                "anthropocene_rl.autonomous_v7.labor_training",
                "pretrain-critic",
                "--checkpoint",
                init,
                "--epochs",
                str(arm.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--max-grad-norm",
                str(args.max_grad_norm),
                *_tail_flag(args),
                "--max-steps",
                str(args.determinism_steps),
                "--stability-log",
                work / f"stability-probe-{repeat}.jsonl",
                "--out",
                target,
            ]
            for directory in args.fit_data:
                command += ["--dataset", directory]
            if args.learner_state_continuations:
                command += ["--learner-state-continuations"]
            command += shlex.split(args.critic_args or "")
            _run(command, log, int(args.threads))
        critic_outputs.append(torch.load(target, map_location="cpu", weights_only=True))
    for key in ("task_critic", "cost_critics"):
        comparisons[f"critic:{key}"] = compare_state_dicts(
            critic_outputs[0][key], critic_outputs[1][key]
        )
    adv_outputs = []
    for repeat in (0, 1):
        target = work / f"advantage-probe-{repeat}.pt"
        if not target.exists():
            command = [
                *_python(),
                "anthropocene_rl.autonomous_v7.labor_advantage",
                "fit",
                "--checkpoint",
                critic_full,
                "--payloads",
                *args.fit_payloads,
                "--out",
                target,
                "--epochs",
                str(args.determinism_advantage_epochs),
                "--seed",
                str(seed),
                "--stability-log",
                work / f"stability-advantage-probe-{repeat}.jsonl",
            ]
            command += shlex.split(args.advantage_args or "")
            _run(command, log, int(args.threads))
        adv_outputs.append(torch.load(target, map_location="cpu", weights_only=True))
    comparisons["advantage:head"] = compare_state_dicts(
        adv_outputs[0]["state_dict"], adv_outputs[1]["state_dict"]
    )
    report = {
        "arm": arm.name,
        "seed": seed,
        "critic_steps": int(args.determinism_steps),
        "advantage_epochs": int(args.determinism_advantage_epochs),
        "threads": int(args.threads),
        "comparisons": comparisons,
        "tolerance": threshold("determinism_tolerance"),
        "note": "CPU; same seed, data, thread count and code; bitwise expected",
    }
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return out


def abort_evidence(work: Path, arm: ArmSpec, seed: int) -> dict[str, Any] | None:
    """A fit aborted by the stability monitor is evidence of a violation (FAIL), not a gap."""
    critic = work / "critic.pt.stability-abort.json"
    advantage = work / "advantage.pt.stability-abort.json"
    if not critic.exists() and not advantage.exists():
        return None

    def load(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    return {
        "schema": EVIDENCE_SCHEMA,
        "arm": arm.name,
        "seed": int(seed),
        "privileged": arm.privileged,
        "aborted": True,
        "stability": {"critic": load(critic), "advantage": load(advantage)},
    }


def command_run(args: argparse.Namespace) -> int:
    arms = parse_arms(args.arms)
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.local and len(seeds) > threshold("local_seeds_max"):
        raise ValueError(f"local runs use at most {threshold('local_seeds_max')} seeds")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    jobs = [(arm, seed) for seed in seeds for arm in arms]
    failures = {}

    candidate = next(a for a in arms if a.role == "candidate")
    determinism_path: Path | None = None

    def one(job):
        nonlocal determinism_path
        arm, seed = job
        try:
            evidence = run_fit(args, arm, seed)
        except Exception as error:  # recorded; the receipt HOLDs (or FAILs an abort)
            failures[f"{arm.name}/seed-{seed}"] = repr(error)
            print(
                f"[r5-acceptance] {arm.name} seed {seed} failed: {error!r}",
                file=sys.stderr,
                flush=True,
            )
            return None
        # The determinism probe needs only this fit's inputs; running it in the same
        # worker overlaps it with the remaining fits instead of adding a serial tail.
        if args.determinism_steps > 0 and arm == candidate and seed == seeds[0]:
            try:
                determinism_path = run_determinism(args, candidate, seeds[0])
            except Exception as error:
                failures["determinism"] = repr(error)
        return evidence

    with ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as pool:
        list(pool.map(one, jobs))
    evidence_paths = []
    for arm, seed in jobs:
        work = Path(args.out) / arm.name / f"seed-{seed}"
        path = work / "evidence.json"
        if not path.exists():
            stub = abort_evidence(work, arm, seed)
            if stub is not None:
                path = work / "evidence-aborted.json"
                path.write_text(json.dumps(stub, indent=1, sort_keys=True), encoding="utf-8")
        evidence_paths.append(path)
    receipt_args = argparse.Namespace(
        evidence=[p for p in evidence_paths if p.exists()],
        arms=args.arms,
        seeds=args.seeds,
        seed_rule=args.seed_rule,
        required_seeds=args.required_seeds,
        determinism=determinism_path,
        attempt_id=args.attempt_id,
        out=Path(args.out) / "gate-receipt.json",
        configuration={
            "arms": [a.state_dict() for a in arms],
            "seeds": seeds,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_grad_norm": args.max_grad_norm,
            "advantage_epochs": args.advantage_epochs,
            "qol_head": args.qol_head,
            "init_world": args.init_world,
            "fit_data": [str(p) for p in args.fit_data],
            "learner_state_continuations": bool(args.learner_state_continuations),
            "calibration_data": str(args.calibration_data),
            "fit_payloads": [str(p) for p in args.fit_payloads],
            "reserved_payloads": [str(p) for p in args.reserved_payloads],
            "control_report": None if args.control_report is None else str(args.control_report),
            "control_report_sha256": file_sha256(args.control_report),
            "jobs": args.jobs,
            "threads": args.threads,
            "smoke": bool(args.smoke),
            "failures": failures,
        },
    )
    return command_receipt(receipt_args)


def command_receipt(args: argparse.Namespace) -> int:
    evidence = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.evidence]
    determinism = (
        None
        if args.determinism is None
        else json.loads(Path(args.determinism).read_text(encoding="utf-8"))
    )
    configuration = getattr(args, "configuration", None) or {
        "evidence": [str(p) for p in args.evidence]
    }
    receipt = build_receipt(
        evidence,
        arms=parse_arms(args.arms),
        seeds=[int(s) for s in args.seeds.split(",")],
        seed_rule=args.seed_rule,
        required_seeds=int(args.required_seeds),
        determinism=determinism,
        configuration=configuration,
        attempt_id=args.attempt_id,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    os.replace(temporary, out)
    manifest = {
        "schema": "labor-r5-evidence-manifest-v1",
        "receipt": {"path": str(out), "sha256": file_sha256(out)},
        "evidence": [{"path": str(p), "sha256": file_sha256(p)} for p in args.evidence],
        "determinism": None
        if args.determinism is None
        else {"path": str(args.determinism), "sha256": file_sha256(args.determinism)},
        "artifacts": receipt["artifacts"],
    }
    (out.parent / "evidence-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(summarize(receipt))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    control = None
    if args.control_report is not None:
        control = json.loads(Path(args.control_report).read_text(encoding="utf-8"))
    timings = (
        None if args.timings is None else json.loads(Path(args.timings).read_text(encoding="utf-8"))
    )
    evidence = evaluate_fit(
        arm=args.arm,
        seed=args.seed,
        checkpoint=args.checkpoint,
        advantage=args.advantage,
        calibration_data=args.calibration_data,
        fit_data=args.fit_data,
        reserved_payloads=args.reserved_payloads,
        control_report=control,
        timings=timings,
        batch_size=args.batch_size,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(evidence, indent=1, sort_keys=True, default=str), encoding="utf-8"
    )
    os.replace(temporary, out)
    checks = fit_checks(evidence)
    print(
        f"{args.arm} seed {args.seed}: {checks['primary_status']} "
        f"(G2C all heads {checks['secondary_status']})"
    )
    return 0


def command_determinism(args: argparse.Namespace) -> int:
    import torch

    a = torch.load(args.left, map_location="cpu", weights_only=True)
    b = torch.load(args.right, map_location="cpu", weights_only=True)
    keys = args.keys.split(",")
    report = {
        "comparisons": {key: compare_state_dicts(a[key], b[key]) for key in keys},
        "left": str(args.left),
        "right": str(args.right),
    }
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["comparisons"], indent=2))
    return 0


def summarize(receipt: Mapping[str, Any]) -> str:
    lines = [f"{receipt['gate_id']} attempt {receipt['attempt_id']}: {receipt['machine_verdict']}"]
    for key, value in receipt["verdict_basis"].items():
        if key != "secondary_gate_note":
            lines.append(f"  {key}: {value}")
    for name, report in receipt["arms"].items():
        gate = report[GATE_ID]
        lines.append(
            f"  arm {name}: {gate['status']} rule={gate['rule']} seeds={gate['seeds']} "
            f"every-seed={gate['passes_in_every_seed']} per-seed={gate['per_seed_status']}"
        )
    for finding in receipt["unresolved_findings"][:30]:
        lines.append(f"  - {finding}")
    return "\n".join(lines)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    evaluate = sub.add_parser("evaluate", help="one fit's evidence JSON")
    evaluate.add_argument("--arm", required=True)
    evaluate.add_argument("--seed", type=int, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--advantage", type=Path, required=True)
    evaluate.add_argument("--calibration-data", type=Path, required=True)
    evaluate.add_argument("--fit-data", type=Path, default=None)
    evaluate.add_argument("--reserved-payloads", type=Path, nargs="+", required=True)
    evaluate.add_argument(
        "--control-report",
        type=Path,
        default=None,
        help="critic3 critic-calibration JSON on calibration worlds 90180-90199",
    )
    evaluate.add_argument("--timings", type=Path, default=None)
    evaluate.add_argument("--batch-size", type=int, default=256)
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.set_defaults(handler=command_evaluate)

    receipt = sub.add_parser("receipt", help="aggregate evidence into gate-receipt.json")
    receipt.add_argument("--evidence", type=Path, nargs="+", required=True)
    receipt.add_argument("--arms", default=None)
    receipt.add_argument("--seeds", required=True)
    receipt.add_argument("--seed-rule", choices=SEED_RULES, default=threshold("seed_rule_default"))
    receipt.add_argument("--required-seeds", type=int, default=threshold("vm_seeds"))
    receipt.add_argument("--determinism", type=Path, default=None)
    receipt.add_argument("--attempt-id", required=True)
    receipt.add_argument("--out", type=Path, required=True)
    receipt.set_defaults(handler=command_receipt)

    determinism = sub.add_parser("determinism", help="compare two checkpoints tensor by tensor")
    determinism.add_argument("--left", type=Path, required=True)
    determinism.add_argument("--right", type=Path, required=True)
    determinism.add_argument("--keys", default="task_critic,cost_critics")
    determinism.add_argument("--out", type=Path, required=True)
    determinism.set_defaults(handler=command_determinism)

    run = sub.add_parser("run", help="arm x seed fits, evaluators and the receipt (resumable)")
    run.add_argument("--normalizer-checkpoint", type=Path, required=True)
    run.add_argument("--privileged-normalizer", type=Path, required=True)
    run.add_argument(
        "--fit-data",
        type=Path,
        nargs="+",
        required=True,
        help="r5 teacher fitting dataset (first) plus optional continuation datasets",
    )
    run.add_argument("--learner-state-continuations", action="store_true", default=False)
    run.add_argument("--calibration-data", type=Path, required=True)
    run.add_argument(
        "--fit-payloads",
        type=Path,
        nargs="+",
        required=True,
        help="search candidate payloads from fitting worlds (A pairs)",
    )
    run.add_argument(
        "--reserved-payloads",
        type=Path,
        nargs="+",
        required=True,
        help="held-out-test candidate payloads of reserved worlds 90154-90159",
    )
    run.add_argument("--control-report", type=Path, default=None)
    run.add_argument(
        "--arms", default=None, help="name:privileged|plain:epochs:candidate|control,..."
    )
    run.add_argument("--seeds", default="0,1,2,3,4")
    run.add_argument("--seed-rule", choices=SEED_RULES, default=threshold("seed_rule_default"))
    run.add_argument("--required-seeds", type=int, default=threshold("vm_seeds"))
    run.add_argument("--local", action="store_true", default=False, help="cap seeds at 3")
    run.add_argument("--init-world", type=int, default=90100)
    run.add_argument("--qol-head", default="logistic,td-lambda,0.8")
    run.add_argument("--init-args", default="", help="extra initialize arguments")
    run.add_argument("--critic-args", default="", help="extra pretrain-critic arguments")
    run.add_argument("--advantage-args", default="", help="extra advantage fit arguments")
    run.add_argument("--batch-size", type=int, default=256)
    run.add_argument("--lr", type=float, default=3e-4)
    run.add_argument("--max-grad-norm", type=float, default=1.0)
    run.add_argument("--advantage-epochs", type=int, default=50)
    run.add_argument("--stability-every", type=int, default=25)
    run.add_argument("--resume-every", type=int, default=100)
    run.add_argument("--determinism-steps", type=int, default=50)
    run.add_argument("--determinism-advantage-epochs", type=int, default=2)
    run.add_argument("--jobs", type=int, default=4)
    run.add_argument("--threads", type=int, default=14)
    run.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help="local mechanics smoke: admit truncated-zero targets; the receipt is forced to HOLD",
    )
    run.add_argument("--attempt-id", required=True)
    run.add_argument("--out", type=Path, required=True)
    run.set_defaults(handler=command_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    return int(args.handler(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
