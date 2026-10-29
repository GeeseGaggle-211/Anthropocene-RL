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
  ``all-seeds`` (every seed passes every check) or
  ``mean-with-floor`` (the seed mean passes every scalar threshold, the worst
  seed is within ``floor_tolerance`` of it, and every non-scalar and every
  stability check passes in every seed). The receipt records the rule used.
* **Arms.** Privileged critic versus the non-privileged control with the same
  data, support, tail and budget; the paired per-seed effect of privilege is
  reported, and a confounded comparison HOLDs.
* **Determinism.** The same seed and data give bitwise-equal parameters (or
  within the documented CPU tolerance, reported as such).
* **Trajectory diagnostics (section 4a, never gate).** Every evidence file
  carries ``trajectory_diagnostics`` (``labor_r5_trajectory``: T1 myopia, T3
  adaptation lag, T4 value tracking when a recorded autonomous trace is given)
  and the receipt aggregates them per arm under ``trajectory_diagnostics``
  (seed means and spreads), optionally compared with a baseline receipt
  (``--trajectory-baseline``): metrics worse by more than their seed spread are
  flagged for review. They never enter the verdict.

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
# Gate versions (user 28 Oct). Evidence without ``gate_version`` is judged by gate v2 (the QoL
# absolute calibration and n-step residual gates), so old evidence and receipts keep their
# verdicts; ``evaluate`` now writes relative-v1 evidence.
GATE_V2 = "gate-v2"
GATE_RELATIVE_V1 = "relative-v1"
GATE_VERSIONS = (GATE_V2, GATE_RELATIVE_V1)
RELATIVE_VARIANTS = ("qol_years", "population_years", "combined")
# User 28 Oct (follow-up): the combined elder reward is the only gating variant of (a) and (b);
# the per-head variants are diagnostics.
RELATIVE_GATED_VARIANTS = ("combined",)

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
    # QoL gate v2 (user 28 Oct): the calibration gate's target and unit. Thresholds
    # (qol_nmae_max, qol_abs_bias_max) are unchanged; only the target and the unit moved.
    "qol_calibration_target": {
        "value": "nstep-k5-bootstrap-v1",
        "status": "approved (user 28 Oct)",
        "source": "nMAE: labor_pretrain_eval.QOL_GATE_TARGET: k = td_nstep_k observed rewards "
        "+ gamma^k x the critic's own bootstrap; terminal windows exact; windows reaching the "
        "collection window's living end are reported, not gated; the Monte Carlo metrics stay "
        "as non-gating diagnostics; ranking unchanged (Monte Carlo)",
    },
    "qol_bias_target": {
        "value": "mc-tailcut-0.5-v1",
        "status": "approved (coordinator amendment of the user's 28 Oct tail-robust target)",
        "source": "|bias|: labor_pretrain_eval.QOL_BIAS_TARGET: Monte Carlo bias on rows whose "
        "tail weight c_i tail_i / G_i <= 0.5 (the n-step target sees a uniform offset c only "
        "as ~0.22 c); the n-step bias is a non-gating diagnostic",
    },
    "qol_bias_min_rows": {
        "value": 200,
        "status": "approved (coordinator, 28 Oct)",
        "source": "a stratum with fewer tail-cut rows HOLDs its bias check",
    },
    "qol_bias_min_worlds": {
        "value": 3,
        "status": "approved (coordinator, 28 Oct)",
        "source": "a stratum with tail-cut rows from fewer worlds HOLDs its bias check",
    },
    "qol_reference_scale": {
        "value": "artifact (--qol-reference-scale); absent: frozen 43.152",
        "status": "approved (user 28 Oct)",
        "source": "std of the per-row Monte Carlo QoL return on the fitting batches "
        "(qol-reference-scale subcommand); the unit of qol:b and of the qol:c TD residual",
    },
    "td_residual_abs_bias_max": {"value": 0.05, "status": "approved (user 26 Oct)", "source": "reference scales (3.1)"},
    "td_residual_mae_max": {"value": 0.20, "status": "approved (user 26 Oct)", "source": "reference scales (3.1)"},
    # Rolling-window n-step residual (r5-b criterion c, replaced one-step TD residual).
    "td_unavailable_max": {
        "value": 0.05,
        "status": "proposed",
        "source": "windows ending at a living cutoff without a declared tail are excluded; above this share HOLD",
    },
    "td_nstep_k": {
        "value": 5,
        "status": "approved (user 26 Oct)",
        "source": "window length in decisions, physical-time discount exp(-elapsed/400y)",
    },
    "td_position_group_width": {
        "value": 10,
        "status": "approved (user 26 Oct)",
        "source": "decision-index groups 1-10, 11-20, ... within each village trajectory",
    },
    "td_position_abs_bias_max": {
        "value": 0.10,
        "status": "approved (user 26 Oct)",
        "source": "per-position-group |bias| in QoL reference scales",
    },
    "td_position_min_windows": {
        "value": 100,
        "status": "approved (user 26 Oct)",
        "source": "groups with fewer windows are reported, not gated",
    },
    "gap_nmae_max": {
        "value": 0.10,
        "status": "approved (user 26 Oct)",
        "source": "cal nMAE - fit nMAE; remedy: wider data (3.2)",
    },
    "gap_ranking_max": {
        "value": 0.10,
        "status": "approved (user 26 Oct)",
        "source": "fit rank - cal rank; remedy: wider data (3.2)",
    },
    "gap_a_ranking_max": {
        "value": 0.10,
        "status": "approved (user 26 Oct)",
        "source": "A held-out - reserved; remedy: wider data (3.2)",
    },
    "seed_rule_default": {
        "value": "mean-with-floor",
        "status": "approved (user 26 Oct)",
        "source": "user-approved default seed rule",
    },
    "floor_tolerance": {"value": 0.02, "status": "approved (user 26 Oct)", "source": "mean-with-floor rule"},
    "determinism_tolerance": {
        "value": 1e-6,
        "status": "approved (user 26 Oct)",
        "source": "max relative CPU drift (3.4)",
    },
    "determinism_tolerance_cuda": {
        "value": 1e-5,
        "status": "approved (user 26 Oct)",
        "source": "labor_device.CUDA_DETERMINISM_TOLERANCE: same-GPU repeat drift, "
        "deterministic cuDNN/cuBLAS, TF32 off; a non-bitwise CUDA result within it PASSes",
    },
    "stability_thresholds": {
        "value": "labor_stability.StabilityThresholds",
        "status": "approved (user 26 Oct)",
        "source": "brief s17; each decision below approved individually",
    },
    "stability_rollbacks": {
        "value": {"max_consecutive_failures": 3, "max_total_rollbacks": 5},
        "status": "approved (user 26 Oct)",
        "source": "2.1: K = 3; more than 5 rollbacks fails the fit. Multi-seed escalation: "
        "a rollback-prone arm across seeds is escalated to the user, not averaged away",
    },
    "stability_dominance": {
        "value": {"window_points": 16, "steps": 400, "share": 0.90, "persistence": 0.75,
                  "min_progress": 0.01},
        "status": "approved (user 26 Oct)",
        "source": "2.2: 16-point window (400 steps)",
    },
    # User decision 29 Oct (60-world scale-out): pretrain-critic --loss-balancing gradnorm.
    # GradNorm equalises the task heads' encoder gradients, so gradient_dominance is advisory
    # for such a fit and task_progress_stall gates instead (mandatory, live early stop, seed
    # gate); it reuses the dominance window, its stalled_progress computation and
    # min_progress. Fits without gradnorm are judged exactly as before.
    "stability_task_progress_stall": {
        "value": {"applies": "pretrain-critic --loss-balancing gradnorm", "window_points": 16,
                  "steps": 400, "min_progress": 0.01, "heads": ["qol_years", "population_years"],
                  "gradient_dominance": "advisory under gradnorm"},
        "status": "approved (user 29 Oct)",
        "source": "scale-out: fixed QoL weights 30/50 tripped gradient_dominance at 60 worlds "
        "(task_encoder at step 1875; global_clip at step 2225); any task head's relative "
        "progress below dominance_min_progress over the dominance window FAILs",
    },
    "stability_saturation": {
        "value": "see saturation_* entries",
        "status": "approved (user 26 Oct)",
        "source": "2.3: relaxed low-side/early-stratum rule",
    },
    "stability_td_trajectory": {
        "value": "see td_* StabilityThresholds fields",
        "status": "approved (user 26 Oct)",
        "source": "2.4: warm-up, diminishing trajectory, divergence, hard [0, W] bound",
    },
    "stability_population_tail": {
        "value": {"upper_excess_alarm": 0.01, "top_bin_mass_alarm": 1e-3},
        "status": "approved (user 26 Oct)",
        "source": "2.5: ESCALATE any trigger to the user",
    },
    # User decision 28 Oct (scale-out Q1): for r5 the population-tail check is reported, not
    # gated. It never fails stability and never trips the seed gate; every other stability
    # check still gates. run/receipt --population-tail gate restores the 26 Oct rule.
    "population_tail_policy_r5": {
        "value": "advisory",
        "status": "approved (user 28 Oct)",
        "source": "scale-out Q1: the G1 pilot's FAIL is the softmax floor over empty bins above "
        "the observed maximum (Fable audit); triggers are still reported and escalated",
    },
    "stability_advantage_overfit": {
        "value": {"overfit_gap": 0.10, "overfit_heldout_drop": 0.03},
        "status": "approved (user 26 Oct)",
        "source": "2.6: standing remedy: more scenarios and heuristic variety; escalate only on "
        "trajectory-knowledge regression (GRU case)",
    },
    "stability_privileged_attention": {
        "value": {"attention_ratio_floor": 0.10},
        "status": "approved (user 26 Oct)",
        "source": "2.7: advisory",
    },
    # TD(lambda) drift: judged on the trajectory (large early drift/error are expected).
    "td_warmup_refreshes": {
        "value": 2,
        "status": "approved-trajectory-rule (user 26 Oct)",
        "source": "first refreshes ignored; drift and error may be large",
    },
    "td_final_ratio": {
        "value": 0.5,
        "status": "approved-trajectory-rule (user 26 Oct)",
        "source": "final drift/error at most 0.5x post-warm-up max; log-slope negative",
    },
    "td_divergence_increase": {
        "value": 0.10,
        "status": "approved-trajectory-rule (user 26 Oct)",
        "source": "mandatory: 3 consecutive post-warm-up increases each >10%",
    },
    "td_drift_floor": {
        "value": 1e-3,
        "status": "approved-trajectory-rule (user 26 Oct)",
        "source": "fraction of W; values below count as converged",
    },
    "td_bound_tolerance": {
        "value": 1e-6,
        "status": "approved-trajectory-rule (user 26 Oct)",
        "source": "mandatory hard [0, W] target bound, relative to W (unchanged)",
    },
    # TD trajectory rule v2 (user 28 Oct; labor_stability TD_RULE_V2). New fits use it; the
    # v1 entries above judge resume states and receipts written before it.
    "td_rule": {
        "value": "v2-floor-or-declining",
        "status": "approved (user 28 Oct)",
        "source": "warm-up by value; PASS declining or stationary-below-floor on the ERROR "
        "series; drift reported with an advisory flag only; old states keep v1",
    },
    "td_warmup_median_factor": {
        "value": 1.5,
        "status": "approved (user 28 Oct)",
        "source": "drop leading refreshes until the first within this x the running median",
    },
    "td_stationary_refreshes": {
        "value": 6,
        "status": "approved (user 28 Oct)",
        "source": "K: stationary window; fewer usable post-warm-up refreshes HOLD",
    },
    "td_stationary_spread": {
        "value": 0.5,
        "status": "approved (user 28 Oct)",
        "source": "stationary: max - min over the last K <= this x their median",
    },
    "td_stationary_error_floor": {
        "value": 0.05,
        "status": "approved (user 28 Oct); recalibrate from the first clean fit",
        "source": "stationary branch: median of the last K error values <= this x W",
    },
    "td_stationary_drift_floor": {
        "value": 0.05,
        "status": "approved (user 28 Oct); recalibrate from the first clean fit",
        "source": "advisory drift series: median of the last K drift values <= this x W",
    },
    "td_divergence_median_factor": {
        "value": 1.5,
        "status": "approved (user 28 Oct)",
        "source": "live alarm: 3 consecutive > td_divergence_increase rises ending above this "
        "x the stationary median (error series; drift advisory); admissible bounds unchanged",
    },
    # Gate relative-v1 (user decision 28 Oct ~16:30Z; labor_pretrain_eval.RELATIVE_DEFINITION).
    "gate_version": {
        "value": GATE_RELATIVE_V1,
        "status": "approved (user 28 Oct)",
        "source": "relative accuracy primary, absolute calibration diagnostic; evidence without "
        "gate_version keeps gate v2",
    },
    "rel_lag_min": {
        "value": 5,
        "status": "approved (user 28 Oct)",
        "source": "gate (a): pairs of one camp trajectory at least 5 decisions apart",
    },
    "rel_margin_w": {
        "value": 0.02,
        "status": "approved (user 28 Oct)",
        "source": "informative pairs: realised discounted-return difference > 0.02 W (QoL)",
    },
    "rel_variant_margins": {
        "value": "population 0.02 W s_pop / s_qol; combined w_qol 0.02 W / s_qol",
        "status": "proposed",
        "source": "the same 0.02 W of QoL expressed through the PPO combination "
        "(labor_pretrain_eval.relative_margins)",
    },
    "rel_variants": {
        "value": list(RELATIVE_GATED_VARIANTS),
        "status": "approved (user 28 Oct, follow-up)",
        "source": "gating variant: the combined elder reward PPO consumes "
        "(labor_ppo.combine_labor_advantages, balanced, unit cost multipliers); QoL and "
        "population-years are reported as non-gating diagnostics",
    },
    "rel_lag_ordering_min": {
        "value": 0.70,
        "status": "approved (user 28 Oct)",
        "source": "gate (a): world-balanced within-trajectory ordering, lag >= 5, margin 0.02 W",
    },
    "rel_all_pairs_ordering_min": {
        "value": 0.65,
        "status": "approved (user 28 Oct, combined variant)",
        "source": "gate (b): world-balanced all-pairs within-trajectory ordering, margin 0.02 W; "
        "measured E5 0.70 in-sample and held out, F1 0.64 held out, untrained init 0.41",
    },
    "rel_min_informative_pairs": {
        "value": 200,
        "status": "proposed",
        "source": "an ordering check with fewer informative pairs HOLDs",
    },
    "rel_min_worlds": {
        "value": 3,
        "status": "proposed",
        "source": "an ordering check with informative pairs from fewer worlds HOLDs",
    },
    "rel_td0_abs_bias_max": {
        "value": 0.02,
        "status": "proposed",
        "source": "gate (c): |world-balanced mean| of V(s_t) - [r_t + gamma_t V(s_t+1)] in QoL "
        "reference scales; measured pilots |bias| <= 0.007",
    },
    "rel_td0_mae_max": {
        "value": 0.12,
        "status": "proposed",
        "source": "gate (c): world-balanced MAE of the one-step residual in QoL reference "
        "scales; measured E5 0.085 in-sample / 0.093 held out, untrained init 0.35",
    },
    # Advantage magnitude on the combined reward (user 28 Oct; NON-GATING for now).
    "adv_sign_agreement_min": {
        "value": {"1": 0.55, "5": 0.65},
        "status": "proposed (NOT recommended for gating: it does not separate an untrained "
        "critic)",
        "source": "headline: world-balanced, world-centred sign agreement of the critic's "
        "k-step combined advantage with A_MC = G - V "
        "(labor_pretrain_eval.ADVANTAGE_MAGNITUDE_DEFINITION), all stratum; advisory flag "
        "only. Measured b04 held out k=1/k=5: E5 0.579/0.684, F1 0.568/0.667, untrained init "
        "0.672/0.723 (raw form E5 0.506/0.598, F1 0.487/0.562, init 0.664/0.706): A_MC shares "
        "the critic's V, so V errors move both sides",
    },
    "adv_slope_range": {
        "value": [0.5, 2.0],
        "status": "proposed",
        "source": "slope of A_MC on A_hat_k (1 for an exact critic); advisory flag only",
    },
    "adv_std_ratio_range": {
        "value": [0.15, 2.0],
        "status": "proposed",
        "source": "std(A_hat_k) / std(A_MC) (below 1 for an exact critic); advisory flag only",
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
        "labor_device.py",
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


POPULATION_TAIL_POLICIES = ("gate", "advisory")
POPULATION_TAIL_CHECK = "population_tail"


def _advisory_checks(population_tail: str) -> tuple[str, ...]:
    if population_tail not in POPULATION_TAIL_POLICIES:
        raise ValueError(f"population-tail policy must be one of {POPULATION_TAIL_POLICIES}")
    return (POPULATION_TAIL_CHECK,) if population_tail == "advisory" else ()


def stability_checks(
    label: str, summary: Mapping[str, Any] | None, *, advisory: Sequence[str] = ()
) -> list[dict[str, Any]]:
    """Mandatory: an admitted stability receipt; a mandatory FAIL fails the fit.

    ``advisory`` names stability checks that are reported (``population_tail_advisory``)
    instead of gated (r5: ``population_tail``, user 28 Oct)."""
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
    # A live early stop (labor_stability EARLY_STOP_CHECKS) FAILs the same check id the
    # end-of-run judgement uses; the row then says where the fit was stopped.
    early = summary.get("early_stop") or {}
    for name, record in sorted((summary.get("checks") or {}).items()):
        if not record.get("mandatory") or record.get("status") == "NOT_APPLICABLE":
            continue
        if name in advisory:
            continue
        row = check(
            f"stability:{label}:{name}",
            "stability",
            record.get("status"),
            "PASS",
            "==",
            status=record.get("status"),
            every_seed=True,
        )
        if early.get("check") == name:
            row["stopped_early_at_step"] = int(early["step"])
            row["early_stop_reason"] = early.get("reason")
        out.append(row)
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
    # (b) bounded-head calibration per stratum. Gate v2 (user 28 Oct): evidence carrying the
    # n-step section is judged against the tail-robust n-step target in its reference-scale
    # unit; older evidence keeps the Monte Carlo metrics it was judged on.
    nstep = _get(calibration, "calibration", "qol_nstep")
    for stratum in strata:
        if nstep is not None:
            metrics = _get(nstep, "strata", stratum)
            cut = None if metrics is None else metrics.get("bias_tailcut")
            detail = {
                "target": _get(nstep, "target", "id"),
                "bias_target": None if cut is None else cut.get("target"),
                "bias_tailcut": cut,
                "nstep_bias_non_gating": None if metrics is None else metrics.get("bias"),
                "scale": _get(nstep, "reference_scale", "value"),
                "scale_source": _get(nstep, "reference_scale", "source"),
                "scale_digest": _get(nstep, "reference_scale", "digest"),
                "rows": None if metrics is None else metrics.get("rows"),
                "monte_carlo_non_gating": None if metrics is None else metrics.get("monte_carlo"),
            }
        else:
            metrics = _head(calibration, "qol_years", stratum)
            detail = None
        nmae = None if metrics is None else metrics.get("nmae")
        bias_status = None
        if nstep is not None:
            # Gate v2: |bias| on the tail-cut Monte Carlo rows; too little coverage HOLDs.
            cut = detail["bias_tailcut"]
            bias = None if cut is None else cut.get("bias")
            if cut is not None and bias is not None and cut.get("coverage") != "PASS":
                bias_status = "HOLD"
        else:
            bias = None if metrics is None else metrics.get("bias")
        out.append(
            check(
                f"qol:b:nmae:{stratum}", "qol", nmae, threshold("qol_nmae_max"), "<=",
                detail=detail,
            )
        )
        out.append(
            check(
                f"qol:b:abs-bias:{stratum}",
                "qol",
                None if bias is None else abs(float(bias)),
                threshold("qol_abs_bias_max"),
                "<=",
                status=bias_status,
                detail=detail,
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
    # (c) n-step residual consistency along each village's own trajectory. More than 5% of
    # windows without a declared tail is missing evidence (HOLD), never a zero-tail pass.
    share = None if td is None else td.get("unavailable_share")
    if td is not None and share is not None and float(share) > threshold("td_unavailable_max"):
        td = {"unavailable_share": share, "bias": None, "mae": None, "groups": []}
        out.append(
            check(
                "qol:c:td-residual-unavailable",
                "qol",
                None,
                threshold("td_unavailable_max"),
                "<=",
                detail={"unavailable_share": share},
            )
        )
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
    # Per-position gate: every group with enough windows needs |bias| <= 0.10.
    for group in [] if td is None else td.get("groups") or []:
        if not group.get("gated"):
            continue
        out.append(
            check(
                f"qol:c:td-residual-position-{group['first']}-{group['last']}",
                "qol",
                abs(float(group["bias"])),
                threshold("td_position_abs_bias_max"),
                "<=",
                detail={"windows": group["windows"], "bias": group["bias"]},
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


FOOD_TRADE_ADVISORY_MARGIN = 0.05


def actor_fidelity_by_family_report(
    actor: Any, dataset: Mapping[str, Any], normalizer: Any, batch_size: int
) -> dict[str, Any]:
    """Report-only actor fidelity per scenario family on the calibration rows."""
    from .labor_pretrain_eval import _actor_arrays, actor_fidelity_by_family

    try:
        arrays = _actor_arrays(actor, dataset, batch_size)
    except (KeyError, ValueError) as error:  # datasets without actor labels
        return {"unavailable": str(error)}
    return actor_fidelity_by_family(arrays, dataset, normalizer)


def food_trade_advisory(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Advisory, never gating (user condition, 26 Oct): with family weights on, flag when
    food-trade villages calibrate or rank worse than all rows by more than 0.05."""
    weighting = evidence.get("family_weighting")
    margin = FOOD_TRADE_ADVISORY_MARGIN
    flags: list[dict[str, Any]] = []
    calibration = _get(evidence, "calibration", "calibration") or {}
    family = (calibration.get("by_scenario_family") or {}).get("food-trade")
    for head, strata in (calibration.get("heads") or {}).items():
        everything = strata.get("all") or {}
        mine = (family or {}).get(head) or {}
        nmae_all, nmae_ft = everything.get("nmae"), mine.get("nmae")
        if nmae_all is not None and nmae_ft is not None:
            flags.append(
                {"check": f"V:{head}:nmae", "all": nmae_all, "food_trade": nmae_ft,
                 "flag": nmae_ft > nmae_all + margin}
            )
        rank_all = (everything.get("ranking") or {}).get("accuracy")
        rank_ft = (mine.get("ranking") or {}).get("accuracy")
        if rank_all is not None and rank_ft is not None:
            flags.append(
                {"check": f"V:{head}:ranking", "all": rank_all, "food_trade": rank_ft,
                 "flag": rank_ft < rank_all - margin}
            )
    advantage = evidence.get("advantage_eval") or {}
    a_family = ((advantage.get("by_scenario_family") or {}).get("food-trade") or {}).get(
        "ranking", {}
    )
    for head in ("qol_years", "population_years", "utility:root"):
        rank_all = ((advantage.get("ranking") or {}).get(head) or {}).get("accuracy")
        rank_ft = (a_family.get(head) or {}).get("accuracy")
        if rank_all is not None and rank_ft is not None:
            flags.append(
                {"check": f"A:{head}:ranking", "all": rank_all, "food_trade": rank_ft,
                 "flag": rank_ft < rank_all - margin}
            )
    if weighting is None:
        status = "NOT-APPLICABLE"
    elif not flags:
        status = "NO-FOOD-TRADE-EVIDENCE"
    else:
        status = "FLAG" if any(f["flag"] for f in flags) else "OK"
    return {
        "check": "advisory:food-trade-vs-all",
        "gating": False,
        "margin": margin,
        "family_weights_on": weighting is not None,
        "status": status,
        "comparisons": flags,
    }


def gate_version(evidence: Mapping[str, Any]) -> str:
    """The gate version an evidence record is judged by (absent: gate v2)."""
    version = evidence.get("gate_version") or GATE_V2
    if version not in GATE_VERSIONS:
        raise ValueError(f"unknown gate version {version!r}")
    return version


def _coverage_status(record: Mapping[str, Any] | None) -> str | None:
    if record is None or record.get("accuracy") is None:
        return None
    enough = int(record.get("informative_pairs") or 0) >= int(
        threshold("rel_min_informative_pairs")
    ) and int(record.get("worlds") or 0) >= int(threshold("rel_min_worlds"))
    return None if enough else "HOLD"


def relative_checks(
    calibration: Mapping[str, Any] | None, td: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    """Gate relative-v1 (a) lag >= 5 ordering, (b) all-pairs ordering, (c) TD(0) residual.

    (a)/(b) judge the ``all`` stratum of every variant on the calibration worlds (the other
    strata are in the detail); thin coverage HOLDs. (c) is the QoL head's one-step residual
    on the calibration worlds in QoL reference scales.
    """
    relative = _get(calibration, "calibration", "relative")
    out = []
    for variant in RELATIVE_GATED_VARIANTS:
        record = _get(relative, "variants", variant)
        for gate_id, section, limit in (
            ("a:lag-ordering", "lag_ge_min", "rel_lag_ordering_min"),
            ("b:all-pairs-ordering", "all_pairs", "rel_all_pairs_ordering_min"),
        ):
            strata = None if record is None else record.get(section)
            everything = None if strata is None else strata.get("all")
            out.append(
                check(
                    f"rel:{gate_id}:{variant}",
                    "relative",
                    None if everything is None else everything.get("accuracy"),
                    threshold(limit),
                    ">=",
                    status=_coverage_status(everything),
                    detail=None
                    if record is None
                    else {
                        "margin": record.get("margin"),
                        "min_lag": relative.get("min_lag") if section == "lag_ge_min" else 1,
                        "strata": strata,
                        "no_margin_non_gating": _get(record, "all_pairs_no_margin", "all"),
                    },
                )
            )
    td0 = None if td is None else td.get("td0")
    out.append(
        check(
            "rel:c:td0-abs-bias",
            "relative",
            None if td0 is None or td0.get("bias") is None else abs(float(td0["bias"])),
            threshold("rel_td0_abs_bias_max"),
            "<=",
            detail=None if td0 is None else {"bias": td0.get("bias"), "rows": td0.get("rows")},
        )
    )
    out.append(
        check(
            "rel:c:td0-mae",
            "relative",
            None if td0 is None else td0.get("mae"),
            threshold("rel_td0_mae_max"),
            "<=",
            detail=None
            if td0 is None
            else {"strata": td0.get("strata"), "constant_reference": td0.get("constant_reference")},
        )
    )
    return out


# Gate-v2 checks that relative-v1 reports as diagnostics (never gating).
RELATIVE_DEMOTED_PREFIXES = ("qol:b:nmae:", "qol:b:abs-bias:", "qol:c:", "gap:qol_years:nmae")


def _demoted(row: Mapping[str, Any]) -> bool:
    return any(row["id"].startswith(prefix) for prefix in RELATIVE_DEMOTED_PREFIXES)


def relative_head_diagnostics(calibration: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The per-head (QoL, population-years) ordering variants, never gating."""
    relative = _get(calibration, "calibration", "relative")
    rows = []
    for variant in RELATIVE_VARIANTS:
        if variant in RELATIVE_GATED_VARIANTS:
            continue
        for gate_id, section, limit in (
            ("a:lag-ordering", "lag_ge_min", "rel_lag_ordering_min"),
            ("b:all-pairs-ordering", "all_pairs", "rel_all_pairs_ordering_min"),
        ):
            everything = _get(relative, "variants", variant, section, "all")
            row = check(
                f"rel:{gate_id}:{variant}",
                "relative-diagnostic",
                None if everything is None else everything.get("accuracy"),
                threshold(limit),
                ">=",
                status=_coverage_status(everything),
                detail=_get(relative, "variants", variant, section),
            )
            rows.append(
                {
                    "id": row["id"],
                    "value": row["value"],
                    "reference_threshold": row["threshold"],
                    "would_have_been": row["status"],
                    "strata": row["detail"],
                    "gating": False,
                }
            )
    return rows


def advantage_magnitude_advisory(calibration: Mapping[str, Any] | None) -> dict[str, Any]:
    """Non-gating: sign agreement (headline), slope and std ratio of the combined k-step
    advantage against the proposed bounds, plus the per-head PopArt statistics."""
    report = _get(calibration, "calibration", "advantage_magnitude")
    if report is None:
        return {"check": "advisory:advantage-magnitude", "gating": False, "status": "NO-EVIDENCE"}
    sign_min = threshold("adv_sign_agreement_min")
    slope_lo, slope_hi = threshold("adv_slope_range")
    ratio_lo, ratio_hi = threshold("adv_std_ratio_range")
    rows = []
    for k, strata in (report.get("ks") or {}).items():
        m = strata.get("all") or {}
        sign = m.get("sign_agreement_world_centred", m.get("sign_agreement"))
        slope, ratio = m.get("slope"), m.get("std_ratio")
        rows.append(
            {
                "k": int(k),
                "sign_agreement_world_centred": sign,
                "sign_agreement_raw": m.get("sign_agreement"),
                "sign_agreement_min": sign_min.get(str(k)),
                "slope": slope,
                "std_ratio": ratio,
                "sign_agreement_by_stratum": {
                    name: s.get("sign_agreement_world_centred", s.get("sign_agreement"))
                    for name, s in strata.items()
                },
                "flag": bool(
                    (sign is not None and sign_min.get(str(k)) is not None
                     and sign < sign_min[str(k)])
                    or (slope is not None and not slope_lo <= slope <= slope_hi)
                    or (ratio is not None and not ratio_lo <= ratio <= ratio_hi)
                ),
            }
        )
    return {
        "check": "advisory:advantage-magnitude",
        "gating": False,
        "status": "FLAG" if any(r["flag"] for r in rows) else "OK",
        "headline": "sign_agreement",
        "rows": rows,
    }


def relative_diagnostics(
    evidence: Mapping[str, Any], demoted: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Never gating: n-step and Monte Carlo QoL nMAE/bias next to the constant-predictor
    reference (a predictor outputting the fitting-set mean QoL return), and the demoted
    gate-v2 rows with the verdict they would have had."""
    nstep = _get(evidence, "calibration", "calibration", "qol_nstep") or {}
    constant = nstep.get("constant_reference") or {}
    table = {}
    for stratum, metrics in (nstep.get("strata") or {}).items():
        reference = (constant.get("strata") or {}).get(stratum) or {}
        table[stratum] = {
            "nstep_nmae": metrics.get("nmae"),
            "nstep_nmae_constant": _get(reference, "nstep", "nmae"),
            "nstep_bias": metrics.get("bias"),
            "nstep_bias_constant": _get(reference, "nstep", "bias"),
            "mc_nmae": _get(metrics, "monte_carlo", "nmae"),
            "mc_nmae_constant": _get(reference, "monte_carlo", "nmae"),
            "mc_bias": _get(metrics, "monte_carlo", "bias"),
            "mc_bias_constant": _get(reference, "monte_carlo", "bias"),
            "tailcut_bias": _get(metrics, "bias_tailcut", "bias"),
        }
    return {
        "gating": False,
        "relative_heads": relative_head_diagnostics(evidence.get("calibration")),
        "advantage_magnitude": _get(
            evidence, "calibration", "calibration", "advantage_magnitude"
        ),
        "constant_reference": {k: constant.get(k) for k in ("value", "source", "definition")},
        "units": "QoL reference scales",
        "qol": table,
        "demoted_gate_v2_checks": [
            {
                "id": row["id"],
                "value": row["value"],
                "former_threshold": row["threshold"],
                "would_have_been": row["status"],
            }
            for row in demoted
        ],
    }


def population_tail_advisory(evidence: Mapping[str, Any], policy: str) -> dict[str, Any]:
    """The population-tail stability check of every fit, reported (never gating when the
    policy is advisory). FLAG on any FAIL: still an escalation item for the user."""
    rows = {}
    for label in ("critic", "advantage"):
        record = _get(evidence, "stability", label, "checks", POPULATION_TAIL_CHECK)
        if record is not None:
            rows[label] = dict(record)
    failed = sorted(k for k, v in rows.items() if v.get("status") == "FAIL")
    return {
        "check": "advisory:population-tail",
        "gating": policy == "gate",
        "policy": policy,
        "status": "NOT-APPLICABLE" if not rows else ("FLAG" if failed else "OK"),
        "failed": failed,
        "fits": rows,
        "escalate": bool(failed),
    }


def fit_checks(
    evidence: Mapping[str, Any],
    *,
    strata: Sequence[str] = STRATA,
    population_tail: str = "gate",
) -> dict[str, Any]:
    """Every check of one (arm, seed) fit, grouped into the primary and secondary gates.

    Gate relative-v1 evidence: the QoL absolute calibration and n-step residual checks and the
    QoL nMAE generalisation gap move to ``diagnostics``; the relative checks join the primary
    gate. Every other check is unchanged. Gate v2 evidence is judged exactly as before."""
    version = gate_version(evidence)
    advisory_stability = _advisory_checks(population_tail)
    primary = []
    primary += stability_checks(
        "critic", _get(evidence, "stability", "critic"), advisory=advisory_stability
    )
    primary += stability_checks(
        "advantage", _get(evidence, "stability", "advantage"), advisory=advisory_stability
    )
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
    diagnostics = None
    if version == GATE_RELATIVE_V1:
        demoted = [row for row in primary if _demoted(row)]
        primary = [row for row in primary if not _demoted(row)]
        primary += relative_checks(evidence.get("calibration"), evidence.get("td_residual"))
        diagnostics = relative_diagnostics(evidence, demoted)
    return {
        "gate_version": version,
        **({} if diagnostics is None else {"diagnostics": diagnostics}),
        "primary": primary,
        "secondary": secondary,
        "primary_status": combine([c["status"] for c in primary]),
        "secondary_status": combine([c["status"] for c in secondary]),
        # Advisory only: never part of primary/secondary and never changes a verdict.
        "advisory": [
            food_trade_advisory(evidence),
            qol_tail_advisory(evidence),
            qol_bias_tail_attribution_advisory(evidence),
            *(
                [advantage_magnitude_advisory(evidence.get("calibration"))]
                if gate_version(evidence) == GATE_RELATIVE_V1
                else []
            ),
            population_diagnostics_advisory(evidence),
            population_tail_advisory(evidence, population_tail),
            *validation_advisories(evidence),
        ],
    }


def _bias_attribution(evidence: Mapping[str, Any]) -> dict[str, Any] | None:
    strata = _get(evidence, "calibration", "calibration", "qol_nstep", "strata")
    if not isinstance(strata, Mapping):
        return None
    return {
        name: _get(metrics, "bias_tailcut", "attribution")
        for name, metrics in strata.items()
        if _get(metrics, "bias_tailcut", "attribution") is not None
    }


def qol_bias_tail_attribution_advisory(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Non-gating (user decision 28 Oct): is a QoL bias verdict driven by tail noise?

    Per stratum, the world-balanced mean schedule factor, the predicted tail-induced bias
    and the residual critic bias (``labor_pretrain_eval.QOL_BIAS_TAIL_ATTRIBUTION``). FLAG
    when any stratum's |predicted| > 0.05 or removing it would flip the |bias| verdict;
    else REPORT; NOT-APPLICABLE for evidence without the gate-v2 section."""
    strata = _bias_attribution(evidence)
    if not strata:
        return {
            "check": "advisory:qol-bias-tail-attribution",
            "gating": False,
            "status": "NOT-APPLICABLE",
        }
    flagged = sorted(name for name, a in strata.items() if a.get("flag"))
    return {
        "check": "advisory:qol-bias-tail-attribution",
        "gating": False,
        "status": "FLAG" if flagged else "REPORT",
        "flagged_strata": flagged,
        "world_factors": _get(evidence, "calibration", "calibration", "qol_nstep", "world_factors"),
        "strata": strata,
    }


def population_diagnostics_advisory(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Non-gating (28 Oct): population-years critic error versus roulette-tail noise.

    From the calibration and fitting reports: per stratum the n-step nMAE/bias, the Monte
    Carlo nMAE, the tail-cut bias with its tail attribution, and the population_tail
    explanation (excess by bin, concentration). REPORT, or NOT-APPLICABLE without it."""
    out: dict[str, Any] = {
        "check": "advisory:population-diagnostics",
        "gating": False,
        "status": "NOT-APPLICABLE",
    }
    for label in ("calibration", "fitting"):
        section = _get(evidence, "calibration", label) or {}
        pop = section.get("population_nstep")
        if pop is None:
            continue
        out["status"] = "REPORT"
        out[label] = {
            "strata": {
                name: {
                    "nstep_nmae": m.get("nmae"),
                    "nstep_bias": m.get("bias"),
                    "monte_carlo_nmae": _get(m, "monte_carlo", "nmae"),
                    "monte_carlo_bias": _get(m, "monte_carlo", "bias"),
                    "tailcut_bias": _get(m, "bias_tailcut", "bias"),
                    "tailcut_rows": _get(m, "bias_tailcut", "rows"),
                    "attribution": _get(m, "bias_tailcut", "attribution"),
                }
                for name, m in (pop.get("strata") or {}).items()
            },
            "world_factors": pop.get("world_factors"),
            "reference_scale_provenance": pop.get("reference_scale_provenance"),
            "tail_explanation": section.get("population_tail_explanation"),
        }
    return out


def qol_tail_advisory(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Advisory, never gating (user decision 27 Oct), reported next to the QoL gates: the
    per-world roulette tail factor of the calibration targets, the share of rows whose
    tail exceeds W, the tail's weight in the targets, and the QoL bias on all rows and on
    rows whose tail weight is at most 0.5 (``labor_pretrain_eval.qol_tail_diagnostics``).
    Its status is REPORT (never FLAG), or NOT-APPLICABLE without randomized-tail data."""
    tail = _get(evidence, "calibration", "calibration", "qol_tail")
    return {
        "check": "advisory:qol-tail-factor",
        "gating": False,
        "status": "NOT-APPLICABLE" if tail is None else "REPORT",
        **({} if tail is None else {"calibration": tail}),
    }


def validation_advisories(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Non-gating: the V critic's per-epoch held-out validation (``--validation-dataset``)
    and its validation_divergence alarm. Absent when the fit ran without validation."""
    record = _get(evidence, "stability", "critic", "validation")
    if record is None:
        return []
    return [
        {
            "check": "advisory:critic-validation-divergence",
            "gating": False,
            "status": record.get("status"),
            "rule": record.get("rule"),
            "alarms": record.get("alarms", []),
            "epochs": [
                {k: e.get(k) for k in ("epoch", "train_loss", "total", "task", "cost")}
                for e in record.get("epochs", [])
            ],
        }
    ]


def scenario_family_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Per-family V calibration, A ranking and actor fidelity of one fit (report only)."""
    calibration = (_get(evidence, "calibration", "calibration") or {}).get(
        "by_scenario_family"
    ) or {}
    advantage = (evidence.get("advantage_eval") or {}).get("by_scenario_family") or {}
    actor = evidence.get("actor_fidelity_by_scenario_family") or {}
    out = {}
    for family in sorted(set(calibration) | set(advantage) | set(actor)):
        heads = calibration.get(family) or {}
        out[family] = {
            "V": {
                head: {
                    "rows": m.get("rows"),
                    "nmae": m.get("nmae"),
                    "bias": m.get("bias"),
                    "ranking": (m.get("ranking") or {}).get("accuracy"),
                }
                for head, m in heads.items()
            },
            "A": {
                name: (r or {}).get("accuracy")
                for name, r in ((advantage.get(family) or {}).get("ranking") or {}).items()
            },
            "A_roots": (advantage.get(family) or {}).get("roots"),
            "actor": actor.get(family) if isinstance(actor, Mapping) else None,
        }
    return out


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
        "rel:a:lag-ordering:qol_years": (
            "calibration", "calibration", "relative", "variants", "qol_years", "lag_ge_min",
            "all", "accuracy",
        ),
        "rel:a:lag-ordering:combined": (
            "calibration", "calibration", "relative", "variants", "combined", "lag_ge_min",
            "all", "accuracy",
        ),
        "rel:c:td0-mae": ("td_residual", "td0", "mae"),
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
    device = str(report.get("device", "cpu"))
    cuda = device.startswith("cuda")
    tolerance = threshold("determinism_tolerance_cuda" if cuda else "determinism_tolerance")
    approved = not cuda or PREREGISTERED["determinism_tolerance_cuda"]["status"].startswith(
        "approved"
    )
    for name, result in sorted(report.get("comparisons", {}).items()):
        if result.get("bitwise"):
            out.append(
                check(
                    f"determinism:{name}", "determinism", "bitwise", "bitwise", "==", detail=result
                )
            )
        else:
            value = result.get("max_relative")
            within = value is not None and float(value) <= float(tolerance)
            out.append(
                check(
                    f"determinism:{name}",
                    "determinism",
                    value,
                    tolerance,
                    "<=",
                    # An unapproved tolerance can FAIL but never PASS (now only a guard).
                    status=None if approved or not within else "HOLD",
                    every_seed=True,
                    detail={
                        "note": f"within the documented {'CUDA' if cuda else 'CPU'} "
                        "tolerance, not bitwise",
                        "device": device,
                        "tolerance_status": PREREGISTERED[
                            "determinism_tolerance_cuda" if cuda else "determinism_tolerance"
                        ]["status"],
                        **result,
                    },
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


def with_control_arm_reference(
    by_arm: Mapping[str, Mapping[int, Mapping[str, Any]]], arms: Sequence[ArmSpec]
) -> tuple[dict[str, dict[int, Mapping[str, Any]]], dict[str, Any]]:
    """The other-head regression reference of every fit (user decision 28 Oct, Q2).

    A fit that carries a control report (``--control-report``, critic3) keeps it. Otherwise
    a candidate fit is compared with the seed-matched evaluation of the run's plain control
    arm of the same epoch budget (same calibration worlds, evaluator and units: its
    ``calibration`` section has the control report's shape). A control-arm fit is its own
    reference (zero deltas: the regression check is about the candidate). A fit without a
    seed-matched control evaluation keeps no reference and its regression check HOLDs.
    Returns the per-arm fits to judge and the reference used per fit."""
    judged: dict[str, dict[int, Mapping[str, Any]]] = {}
    used: dict[str, Any] = {}
    for arm in arms:
        reference = next(
            (c for c in arms if c.role == "control" and c.epochs == arm.epochs), None
        )
        out: dict[int, Mapping[str, Any]] = {}
        for seed, record in (by_arm.get(arm.name) or {}).items():
            key = f"{arm.name}/seed-{seed}"
            if record.get("aborted") or record.get("control") is not None:
                out[seed] = record
                used[key] = (
                    None if record.get("aborted") else {"kind": "control-report"}
                )
                continue
            if arm.role == "control":
                source = record
                used[key] = {"kind": "self (reference arm)", "arm": arm.name, "seed": seed}
            else:
                source = (by_arm.get(reference.name) or {}).get(seed) if reference else None
                used[key] = (
                    None
                    if source is None or source.get("calibration") is None
                    else {"kind": "control-arm", "arm": reference.name, "seed": seed}
                )
            if source is None or source.get("calibration") is None:
                out[seed] = record
                continue
            out[seed] = {**record, "control": source["calibration"]}
        judged[arm.name] = out
    return judged, used


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
    trajectory_baseline: Mapping[str, Any] | None = None,
    population_tail: str = "gate",
) -> dict[str, Any]:
    """The gate receipt (brief section 4 fields) with the machine verdict.

    ``population_tail`` ``advisory`` (r5, user 28 Oct) reports the population-tail stability
    check under each fit's advisories instead of gating it.

    ``trajectory_diagnostics`` (section 4a) is reported beside the verdict and never
    enters it; ``trajectory_baseline`` (an earlier receipt) adds the flagged comparison.
    """
    by_arm: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for record in evidence:
        by_arm[record["arm"]][int(record["seed"])] = record
    judged, regression_reference = with_control_arm_reference(by_arm, arms)
    arm_reports = {}
    for arm in arms:
        fits = judged.get(arm.name, {})
        per_seed_primary, per_seed_secondary = {}, {}
        per_seed_advisory: dict[int, Any] = {}
        per_seed_diagnostics: dict[int, Any] = {}
        per_seed_families: dict[int, Any] = {}
        for seed in seeds:
            if seed in fits:
                checks = fit_checks(fits[seed], strata=strata, population_tail=population_tail)
                per_seed_primary[seed] = checks["primary"]
                per_seed_secondary[seed] = checks["secondary"]
                per_seed_advisory[seed] = checks["advisory"]
                if "diagnostics" in checks:
                    per_seed_diagnostics[seed] = checks["diagnostics"]
                per_seed_families[seed] = scenario_family_summary(fits[seed])
            else:
                missing = [check("fit:missing", "fit", None, "present", "==", every_seed=True)]
                per_seed_primary[seed] = missing
                per_seed_secondary[seed] = missing
        arm_reports[arm.name] = {
            "arm": arm.state_dict(),
            "advisory": {str(s): v for s, v in per_seed_advisory.items()},
            "advisory_flagged": any(
                a["status"] == "FLAG" for v in per_seed_advisory.values() for a in v
            ),
            "by_scenario_family": {str(s): v for s, v in per_seed_families.items()},
            **(
                {"diagnostics": {str(s): v for s, v in per_seed_diagnostics.items()}}
                if per_seed_diagnostics
                else {}
            ),
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
    qol_gate = qol_gate_section(evidence)
    versions = sorted({gate_version(r) for r in evidence if not r.get("aborted")})
    versions_status = "PASS" if len(versions) <= 1 else "HOLD"
    smoke = bool(configuration.get("smoke"))
    verdict = combine(
        [gate_status, determinism_status, comparison_status, exclusions["status"]]
        + (["HOLD"] if smoke else [])
        + ([] if qol_gate["status"] == "PASS" else [qol_gate["status"]])
        + ([] if versions_status == "PASS" else [versions_status])
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
    if versions_status != "PASS":
        unresolved.append(
            {"check": "gate-version-consistent", "status": versions_status, "versions": versions}
        )
    if qol_gate["status"] != "PASS":
        unresolved.append(
            {
                "check": "qol:reference-scale-consistent",
                "status": qol_gate["status"],
                "scales": qol_gate["distinct"],
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
        # Gate version(s) the fits were judged by (evidence without one: gate v2).
        "gate_versions": versions,
        # QoL gate v2 (user 28 Oct): the target definition and the reference scale (value,
        # source, digest) each fit's QoL calibration and TD residual were judged in.
        "qol_gate": qol_gate,
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
        "trajectory_diagnostics": _trajectory_section(
            evidence, [a.name for a in arms], seeds, trajectory_baseline
        ),
        "population_tail_policy": population_tail,
        # Other-head regression reference per fit (user 28 Oct, Q2): a supplied critic3
        # control report, else the run's own seed-matched plain control arm.
        "regression_reference": regression_reference,
        "machine_state": verdict,
        "machine_verdict": verdict,
        "verdict_basis": {
            "candidate_arm": primary_arm.name,
            GATE_ID: gate_status,
            "arm_comparison": comparison_status,
            "determinism": determinism_status,
            "historical_fitting_exclusions": exclusions["status"],
            "qol_reference_scale": qol_gate["status"],
            "gate_versions": versions,
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


def _trajectory_section(
    evidence: Sequence[Mapping[str, Any]],
    arms: Sequence[str],
    seeds: Sequence[int],
    baseline: Mapping[str, Any] | None,
) -> dict[str, Any]:
    from .labor_r5_trajectory import receipt_section

    try:
        return receipt_section(evidence, arms=arms, seeds=seeds, baseline=baseline)
    except Exception as error:  # non-gating: a broken diagnostic never blocks the receipt
        return {"gating": False, "error": repr(error)}


def qol_gate_section(evidence: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The QoL gate's target and reference scale per fit (gate v2, user 28 Oct).

    Evidence written before gate v2 carries neither: it is listed as ``monte-carlo`` /
    ``frozen-g0`` (what it was judged on). Fits judged in different units cannot share a
    seed rule or an arm comparison: distinct (value, digest) pairs HOLD the receipt.
    """
    fits: dict[str, Any] = {}
    distinct: dict[str, dict[str, Any]] = {}
    for record in evidence:
        reference = record.get("qol_reference_scale")
        target = record.get("qol_gate_target")
        key = f"{record.get('arm')}/seed-{record.get('seed')}"
        if record.get("aborted"):
            fits[key] = {"aborted": True}
            continue
        scale = (
            {"value": 43.152, "source": "frozen-g0", "digest": None}
            if reference is None
            else {k: reference.get(k) for k in ("value", "source", "digest", "path", "sha256")}
        )
        if target is None:
            named: Any = {"nmae": "monte-carlo (pre gate v2)", "bias": "monte-carlo (pre gate v2)"}
        elif "nmae" in target:
            named = {metric: (target.get(metric) or {}).get("id") for metric in ("nmae", "bias")}
        else:
            named = {"nmae": target.get("id"), "bias": target.get("id")}
        fits[key] = {"target": named, "scale": scale}
        attribution = _bias_attribution(record)
        if attribution:
            fits[key]["bias_tail_attribution"] = {
                "world_factors": _get(
                    record, "calibration", "calibration", "qol_nstep", "world_factors"
                ),
                "strata": attribution,
            }
        distinct[json_digest([scale["value"], scale["digest"]])] = {
            "value": scale["value"],
            "digest": scale["digest"],
            "source": scale["source"],
        }
    targets = {
        json_digest(r.get("qol_gate_target")): r.get("qol_gate_target")
        for r in evidence
        if r.get("qol_gate_target") is not None
    }
    return {
        "definitions": list(targets.values()),
        "fits": fits,
        "distinct": list(distinct.values()),
        "status": "PASS" if len(distinct) <= 1 else "HOLD",
    }


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


def nstep_residuals(
    values: Any,
    reward: Any,
    elapsed: Any,
    terminal: Any,
    turns: Any,
    worlds: Any,
    camps: Any,
    trajectories: Any,
    codes: Any = None,
    tails: Any = None,
    *,
    k: int = 5,
    discount_years: float = 400.0,
    zero_tail_declared: bool = False,
) -> tuple[Any, Any, Any]:
    """Rolling n-step residuals along each trajectory, in the units of ``values``/``reward``.

    residual_t = V(s_t) - [sum_{j<k} (prod_{m<j} g_m) r_{t+j} + (prod_{m<k} g_m) V(s_{t+k})]
    with g_m = exp(-elapsed_m / discount_years), sliding one decision at a time. A window
    that reaches a terminal row uses realized rewards through it and a zero tail; one that
    reaches the trajectory's living end uses ``tails`` at the last row when ``codes``
    declares an exact/randomized tail. A living end with no declared tail is never closed
    with zero: that window is *unavailable* (excluded), unless the dataset's declared method
    is the truncated-zero window (``zero_tail_declared``). Trajectories are keyed (world,
    camp, row_trajectory) and ordered by ``row_turn``. Returns (residual, position,
    available) where ``position`` is the 0-based decision index within the trajectory.
    """
    import numpy as np

    from .labor_bc import TAIL_RANDOMIZED, TAIL_TERMINAL_EXACT

    n = len(values)
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i in range(n):
        groups[(int(worlds[i]), int(camps[i]), int(trajectories[i]))].append(i)
    residual = np.zeros(n)
    position = np.zeros(n, dtype=np.int64)
    available = np.ones(n, dtype=bool)
    for indices in groups.values():
        indices.sort(key=lambda i: turns[i])
        size = len(indices)
        for pos, i in enumerate(indices):
            position[i] = pos
            total, discount = 0.0, 1.0
            tail_value = None
            for j in range(k):
                if pos + j >= size:
                    break
                row = indices[pos + j]
                total += discount * float(reward[row])
                discount *= math.exp(-float(elapsed[row]) / discount_years)
                if terminal[row]:
                    tail_value = 0.0
                    break
                if pos + j + 1 == size:  # living end: stored tail
                    exact = codes is not None and codes[row] in (
                        TAIL_TERMINAL_EXACT,
                        TAIL_RANDOMIZED,
                    )
                    if exact:
                        tail_value = float(tails[row])
                    elif zero_tail_declared:
                        tail_value = 0.0
                    else:
                        available[i] = False
                        tail_value = 0.0
                    break
            else:
                tail_value = float(values[indices[pos + k]])
            residual[i] = float(values[i]) - (total + discount * tail_value)
    return residual, position, available


def position_group_report(
    normalized: Any,
    position: Any,
    weights: Any,
    *,
    width: int,
    min_windows: int,
) -> list[dict[str, Any]]:
    """Per-decision-index-group bias/MAE (groups 1-width, width+1-2 width, ...)."""
    import numpy as np

    groups = []
    if not len(position):
        return groups
    index = np.asarray(position) // int(width)
    for g in sorted(set(index.tolist())):
        mask = index == g
        w = np.asarray(weights)[mask]
        total = float(w.sum())
        count = int(mask.sum())
        if total <= 0:
            continue
        bias = float((np.asarray(normalized)[mask] * w).sum() / total)
        mae = float((np.abs(np.asarray(normalized)[mask]) * w).sum() / total)
        groups.append(
            {
                "group": int(g),
                "first": int(g) * width + 1,
                "last": (int(g) + 1) * width,
                "windows": count,
                "bias": bias,
                "mae": mae,
                "gated": count >= min_windows,
            }
        )
    return groups


def td0_residual_report(
    values: Any,
    dataset: Mapping[str, Any],
    keep: Any,
    scale: float,
    *,
    constant: float | None = None,
) -> dict[str, Any]:
    """Gate relative-v1 (c): the QoL head's one-step (TD(0)) residual along each trajectory.

    residual_t = V(s_t) - [r_t + gamma_t V(s_t+1)] (``labor_pretrain_eval.nstep_targets``
    with k = 1: own bootstrap; a terminal row closes exactly with r_t); the trajectory's
    living-end row (no successor) is excluded, so no Monte Carlo tail enters. World-balanced
    bias and MAE over admitted rows in QoL reference scales, with early/late strata and, with
    ``constant``, the same numbers for a constant predictor (reference, non-gating).
    """
    import numpy as np

    from .labor_bc import CRITIC_DISCOUNT_YEARS, TRAJECTORY_FIELD, elapsed_years
    from .labor_pretrain_eval import (
        NSTEP_GATED_CLOSURES,
        _np,
        decision_index,
        nstep_targets,
        world_balanced,
    )

    n = int(dataset["row_world"].shape[0])
    worlds = _np(dataset, "row_world").reshape(-1)
    turns = _np(dataset, "row_turn").reshape(-1)

    def residuals(v: Any) -> tuple[Any, Any]:
        result = nstep_targets(
            v,
            _np(dataset, "reward_qol_years").reshape(-1),
            elapsed_years(dataset).reshape(-1).double().numpy(),
            _np(dataset, "transition_terminal").reshape(-1),
            turns,
            worlds,
            _np(dataset, "row_camp").reshape(-1),
            _np(dataset, TRAJECTORY_FIELD).reshape(-1)
            if TRAJECTORY_FIELD in dataset
            else np.full(n, -1),
            k=1,
            discount_years=CRITIC_DISCOUNT_YEARS,
        )
        return np.asarray(v, dtype=np.float64) - result["target"], result["closure"]

    delta, closure = residuals(np.asarray(values, dtype=np.float64))
    keep = np.asarray(keep, dtype=bool)
    rows = np.flatnonzero(keep & np.isin(closure, NSTEP_GATED_CLOSURES))
    weights = world_balanced(worlds[rows], np.ones(len(rows), dtype=bool))

    def mean(x: Any, mask: Any = None) -> float | None:
        w = weights if mask is None else weights * mask
        total = float(w.sum())
        return None if total <= 0 else float((x * w).sum() / total)

    normalized = delta[rows] / float(scale)
    early = None
    if len(rows):
        index, count = decision_index(worlds[rows], turns[rows])
        early = index < (count + 1) // 2
    report: dict[str, Any] = {
        "definition": "V(s_t) - [r_t + gamma_t V(s_t+1)], own bootstrap, living-end rows "
        "excluded, terminal rows exact",
        "rows": len(rows),
        "excluded_living_end_rows": int((keep & ~np.isin(closure, NSTEP_GATED_CLOSURES)).sum()),
        "worlds": len(np.unique(worlds[rows])),
        "bias": mean(normalized),
        "mae": mean(np.abs(normalized)),
        "strata": None
        if early is None
        else {
            name: {"bias": mean(normalized, mask), "mae": mean(np.abs(normalized), mask)}
            for name, mask in (("early_turns", early), ("late_turns", ~early))
        },
    }
    if constant is not None:
        c_delta, _ = residuals(np.full(n, float(constant)))
        c_norm = c_delta[rows] / float(scale)
        report["constant_reference"] = {
            "value": float(constant),
            "bias": mean(c_norm),
            "mae": mean(np.abs(c_norm)),
            "gating": False,
        }
    return report


def td_residual_report(
    task: Any,
    costs: Any,
    dataset: Mapping[str, Any],
    batch_size: int = 256,
    qol_reference: Mapping[str, Any] | None = None,
    constant: float | None = None,
) -> dict[str, Any]:
    """Rolling-window n-step consistency of the QoL head along each village's trajectory.

    k-decision windows (``td_nstep_k``) with physical-time discounting; see
    :func:`nstep_residuals`. Units are QoL reference scales. Aggregate bias/MAE are
    world-balanced over admitted rows; ``groups`` holds the per-position bias.
    ``qol_reference`` (``labor_pretrain_eval.load_qol_reference_scale``) is the unit; absent,
    the frozen 43.152 (``scale_source`` records which).
    """
    import numpy as np

    from .labor_bc import (
        RETURN_TAIL_RANDOMIZED,
        TAIL_METHOD_FIELD,
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
    k = int(threshold("td_nstep_k"))
    delta, position, window_ok = nstep_residuals(
        values,
        reward,
        elapsed,
        terminal,
        turns,
        worlds,
        camps,
        trajectories,
        codes,
        tails,
        k=k,
        zero_tail_declared=not randomized,
    )
    scale = (
        LABOR_OBJECTIVE_REFERENCE_SCALES["qol_years"]
        if qol_reference is None
        else float(qol_reference["value"])
    )
    keep = ~_np(dataset, "row_quarantined").reshape(-1).astype(bool) & critic_target_available(
        dataset
    ).numpy().astype(bool)
    admitted = int(keep.sum())
    unavailable = int((keep & ~window_ok).sum())
    rows = np.flatnonzero(keep & window_ok)
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
        "scale_source": "frozen-g0" if qol_reference is None else qol_reference.get("source"),
        "scale_digest": None if qol_reference is None else qol_reference.get("digest"),
        "units": "QoL reference scales",
        "window_decisions": k,
        "admitted_windows": admitted,
        "unavailable": unavailable,
        "unavailable_share": unavailable / admitted if admitted else None,
        "bias": mean(normalized) if total > 0 else None,
        "mae": mean(np.abs(normalized)) if total > 0 else None,
        "groups": position_group_report(
            normalized,
            position[rows],
            weights,
            width=int(threshold("td_position_group_width")),
            min_windows=int(threshold("td_position_min_windows")),
        ),
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
        # Gate relative-v1 (c): the one-step residual on the same values and rows.
        "td0": td0_residual_report(values, dataset, keep, scale, constant=constant),
    }


def _stability_from_checkpoint(state: Mapping[str, Any]) -> Mapping[str, Any] | None:
    return _get(state, "heuristic_critic_pretraining", "stability")


def evaluate_fit(
    *,
    arm: str,
    seed: int,
    checkpoint: Path,
    advantage: Path | None,
    calibration_data: Path | Sequence[Path],
    fit_data: Path | Sequence[Path] | None,
    reserved_payloads: Sequence[Path] | None,
    control_report: Mapping[str, Any] | None = None,
    privileged: bool | None = None,
    timings: Mapping[str, float] | None = None,
    batch_size: int = 256,
    trajectory_datasets: Sequence[Path] = (),
    trajectory_payloads: Sequence[tuple[str, Sequence[Path]]] = (),
    trajectory_traces: Sequence[Path] = (),
    device: str = "cpu",
    scenario_calibration_data: Path | None = None,
    qol_reference_scale: Path | None = None,
    corrections_override: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    """One (arm, seed) fit's evidence: calibration, TD residual, A on reserved worlds, stability.

    ``advantage`` and ``reserved_payloads`` both None is the critic stage (``run --stage
    critic``): the V evidence only (calibration, relative ordering, TD residual, V
    stability), ``stage: critic``; every A-dependent field is None and the checks that read
    them HOLD (``CRITIC_STAGE_DEFERRED_PREFIXES``). Giving exactly one of the two is refused.

    ``qol_reference_scale`` (a ``qol-reference-scale`` artifact) is the unit of the QoL
    calibration gate and of the TD residual; absent, the frozen 43.152. The evidence
    records the record used (value, source, digest, file sha256) and the gate's target.

    ``calibration_data`` and ``fit_data`` are one dataset or several collection batches,
    read as one dataset (``labor_teacher_data.merge_labor_datasets``: world order, equal
    mass per world); a single path is read exactly as before.

    ``device`` (labor_device) runs the stored-data forward passes on a GPU; every
    metric is computed from host copies of the predictions.

    ``trajectory_*`` feed the non-gating section 4a diagnostics: extra critic
    datasets (teacher, DAgger and learner-state continuation data, T3), labelled
    candidate payload sets (for example one per DAgger round, T3) and autonomous
    reports recorded with ``--record-critic-trace`` (T4).

    ``scenario_calibration_data`` (labor_scenarios calibration scenario set) adds the
    report-only per-family calibration under ``scenario_calibration``; per-family A
    ranking on reserved scenario roots is ``advantage_eval.by_scenario_family``.
    """
    import torch

    from .labor_advantage import load_advantage_head, load_candidate_shards
    from .labor_advantage_eval import evaluate_advantage
    from .labor_device import setup
    from .labor_pretrain_eval import (
        QOL_GATE_TARGETS,
        _load_dataset,
        _load_models,
        dataset_label,
        load_qol_reference_scale,
        run_critic_calibration,
    )
    from .labor_training import _normalization_digest
    from .labor_value_calibration import labor_critic_digest

    critic_stage = advantage is None
    if critic_stage != (not reserved_payloads):
        raise ValueError(
            "the A head and the reserved payloads go together: give both (full evidence) or "
            "neither (critic stage)"
        )
    reserved_payloads = list(reserved_payloads or [])
    started = time.time()
    qol_reference = load_qol_reference_scale(qol_reference_scale)
    target = setup(device)
    state, actor, task, costs, normalizer = _load_models(checkpoint)
    task.to(target)  # the actor stays on CPU (only its parameter digest is read)
    costs.to(target)
    calibration = run_critic_calibration(
        checkpoint,
        calibration_data,
        fit_data,
        batch_size,
        device=None if target.type == "cpu" else target,
        qol_reference=qol_reference,
        corrections_override=corrections_override,
    )
    calibration["critic"] = {
        "qol_head": None if task.qol_head is None else task.qol_head.state_dict(),
        "value_support": task.support.state_dict(),
        "privileged": task.privileged_layout is not None,
    }
    from .labor_pretrain_eval import resolve_loader_corrections

    corrections, _ = resolve_loader_corrections(state, corrections_override)
    dataset = _load_dataset(
        calibration_data,
        normalizer,
        require_privileged=task.privileged_layout is not None,
        corrections=corrections,
    )
    td = td_residual_report(
        task,
        costs,
        dataset,
        batch_size,
        qol_reference=qol_reference,
        constant=_get(
            calibration, "calibration", "qol_nstep", "constant_reference", "value"
        ),
    )
    adv_state: dict[str, Any] = {}
    head = adv_config = reserved = adv_eval = None
    adv_fit: dict[str, Any] = {}
    if not critic_stage:
        adv_state = torch.load(advantage, map_location="cpu", weights_only=True)
        for module in (task, costs):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        head, adv_config = load_advantage_head(adv_state, task, costs)
        head.to(target)
        reserved = load_candidate_shards(list(reserved_payloads))
        # Report-only per-family A ranking: state families (food-trade guard included) of
        # every reserved root, added to any curated scenario families it already carries.
        from .labor_scenarios import root_state_families

        for root, families in zip(
            reserved.roots, root_state_families(reserved, normalizer), strict=True
        ):
            root.families = tuple(sorted(set(root.families) | set(families)))
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
    scenario_calibration = None
    if scenario_calibration_data is not None:
        report = run_critic_calibration(
            checkpoint,
            scenario_calibration_data,
            None,
            batch_size,
            device=None if target.type == "cpu" else target,
            qol_reference=qol_reference,
        )
        scenario_calibration = {
            "dataset": str(scenario_calibration_data),
            "dataset_digest": report["calibration"]["dataset_digest"],
            "rows": report["calibration"]["rows"],
            "worlds": report["calibration"]["worlds"],
            "all": {
                name: strata.get("all") for name, strata in report["calibration"]["heads"].items()
            },
            "by_scenario_family": report["calibration"].get("by_scenario_family", {}),
            "gated": False,
        }
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
        "advantage_config": None
        if adv_config is None
        else adv_config.state_dict() | {"seed": None},
        "payload_digests": sorted(adv_fit.get("payload_digests", []) or []),
    }
    trajectory = (
        {"gating": False, "skipped": "critic stage: T1-T4 need the A head and payloads"}
        if critic_stage
        else _trajectory_diagnostics(
            task=task,
            costs=costs,
            head=head,
            normalizer=normalizer,
            dataset=dataset,
            reserved=reserved,
            reserved_payloads=reserved_payloads,
            datasets=trajectory_datasets,
            payloads=trajectory_payloads,
            traces=trajectory_traces,
            batch_size=batch_size,
        )
    )
    timings = dict(timings or {})
    timings["evaluate"] = time.time() - started
    if target.type != "cpu":
        timings["evaluate_device"] = str(target)
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        # Gate relative-v1 (user 28 Oct); evidence written before it carries no field (v2).
        "gate_version": GATE_RELATIVE_V1,
        # "critic": V evidence only (run --stage critic); "all": V plus the A head.
        "stage": "critic" if critic_stage else "all",
        "loader_corrections": corrections,
        "arm": arm,
        "seed": int(seed),
        "privileged": task.privileged_layout is not None
        if privileged is None
        else bool(privileged),
        "artifacts": {
            "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint)},
            "advantage": None
            if critic_stage
            else {"path": str(advantage), "sha256": file_sha256(advantage)},
            "calibration_dataset": {
                "path": dataset_label(calibration_data),
                "digest": dataset["collection"]["dataset_digest"],
            },
            "fit_dataset": None if fit_data is None else dataset_label(fit_data),
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
        "scenario_calibration": scenario_calibration,
        "family_weighting": critic_summary.get("family_weighting"),
        "actor_fidelity_by_scenario_family": actor_fidelity_by_family_report(
            actor, dataset, normalizer, batch_size
        ),
        "td_residual": td,
        "qol_gate_target": QOL_GATE_TARGETS,
        "qol_reference_scale": qol_reference,
        "advantage_fit": None
        if critic_stage
        else {k: v for k, v in adv_fit.items() if k not in ("stability", "loss_history")},
        "advantage_eval": adv_eval,
        "control": control_report,
        "stability": {
            "critic": _stability_from_checkpoint(state),
            "advantage": adv_fit.get("stability"),
        },
        "timings_seconds": timings,
        "trajectory_diagnostics": trajectory,
    }
    return evidence


def _trajectory_diagnostics(
    *,
    task: Any,
    costs: Any,
    head: Any,
    normalizer: Any,
    dataset: Mapping[str, Any],
    reserved: Any,
    reserved_payloads: Sequence[Path],
    datasets: Sequence[Path],
    payloads: Sequence[tuple[str, Sequence[Path]]],
    traces: Sequence[Path],
    batch_size: int,
) -> dict[str, Any]:
    """Section 4a T1/T3/T4 for one fit; failures are recorded, never raised (non-gating)."""
    from .labor_advantage import load_candidate_shards
    from .labor_pretrain_eval import _load_dataset
    from .labor_r5_trajectory import fit_diagnostics, traces_from_reports
    from .labor_value_calibration import labor_critic_digest

    started = time.time()
    inputs: dict[str, Any] = {
        "datasets": [str(p) for p in datasets],
        "payloads": {label: [str(p) for p in paths] for label, paths in payloads},
        "traces": [{"path": str(p), "sha256": file_sha256(p)} for p in traces],
    }
    errors: dict[str, str] = {}
    extra, payload_sets, loaded_traces = [], [], []
    privileged = task.privileged_layout is not None
    for path in datasets:
        try:
            extra.append(
                (str(path), _load_dataset(Path(path), normalizer, require_privileged=privileged))
            )
        except Exception as error:
            errors[f"dataset:{path}"] = repr(error)
    for label, paths in payloads:
        try:
            candidates = load_candidate_shards(list(paths))
            sealed = sorted(set(candidates.worlds) & set(SEALED_WORLDS))
            if sealed:
                raise ValueError(f"sealed worlds {sealed[:5]} are never used")
            payload_sets.append((str(label), candidates))
        except Exception as error:
            errors[f"payload:{label}"] = repr(error)
    for path in traces:
        try:
            loaded_traces.extend(
                traces_from_reports([json.loads(Path(path).read_text(encoding="utf-8"))])
            )
        except Exception as error:
            errors[f"trace:{path}"] = repr(error)
    try:
        report = fit_diagnostics(
            task=task,
            costs=costs,
            head=head,
            calibration_dataset=dataset,
            reserved=reserved,
            reserved_paths=list(reserved_payloads),
            extra_datasets=extra,
            payload_sets=payload_sets,
            traces=loaded_traces,
            critic_digest=labor_critic_digest(task, costs),
            batch_size=batch_size,
        )
    except Exception as error:  # never stops the evidence
        report = {"error": repr(error), "gating": False}
    report["inputs"] = inputs
    report["load_errors"] = errors
    report["seconds"] = time.time() - started
    return report


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


def _device_flag(args: argparse.Namespace) -> list[str]:
    """``--device`` for the stored-data stages; absent on CPU (historical command lines)."""
    device = str(getattr(args, "device", "cpu"))
    return [] if device == "cpu" else ["--device", device]


def _tail_flag(args: argparse.Namespace) -> list[str]:
    """r5 freeze: every critic target uses the randomized tail; only --smoke relaxes it (HOLD)."""
    return [] if getattr(args, "smoke", False) else ["--require-randomized-tail"]


def _early_stop_flag(args: argparse.Namespace) -> list[str]:
    """Live early stop on would-FAIL stability conditions (on in ``run`` unless disabled)."""
    return ["--stability-early-stop"] if getattr(args, "stability_early_stop", False) else []


def _validation_flag(args: argparse.Namespace) -> list[Any]:
    """V critic per-epoch held-out validation on the calibration datasets (monitoring only;
    the calibration worlds are evaluation-only and never reach the training loader)."""
    if not getattr(args, "critic_validation", False):
        return []
    return ["--validation-dataset", *args.calibration_data]


def require_calibration_not_trained(args: argparse.Namespace) -> None:
    """Calibration directories are evaluation-only: never a --fit-data directory."""
    fit = {Path(p).resolve() for p in args.fit_data}
    shared = sorted(str(p) for p in args.calibration_data if Path(p).resolve() in fit)
    if shared:
        raise ValueError(f"calibration data listed as fitting data: {shared}")


def _evaluation_fit_data(args: argparse.Namespace) -> list[Path]:
    """The fitting data of the generalisation gap: every --fit-data directory when they are
    all teacher batches; with continuation or curated scenario sources, the first one (the
    teacher dataset, as before)."""
    gap = getattr(args, "gap_fit_data", None)
    if gap:
        return list(gap)
    mixed = bool(args.learner_state_continuations) or bool(
        getattr(args, "curated_scenarios", False)
    )
    return [args.fit_data[0]] if mixed else list(args.fit_data)


STAGES = ("all", "critic")


def _stage(args: argparse.Namespace) -> str:
    return str(getattr(args, "stage", "all") or "all")


def _population_tail(args: argparse.Namespace) -> str:
    """run/receipt --population-tail (default advisory for r5, user 28 Oct); a namespace
    without the option (library callers) keeps the gating rule."""
    return str(getattr(args, "population_tail", None) or "gate")


def evidence_name(stage: str) -> str:
    """Per-fit evidence file of a stage: the critic stage never shadows the full evidence,
    so a later ``--stage all`` run in the same OUT resumes at the A fit."""
    return "evidence-critic.json" if stage == "critic" else "evidence.json"


def require_run_inputs(args: argparse.Namespace) -> None:
    """``--stage all`` needs both payload sets; ``--gap-fit-data`` must be fitting data."""
    stage = _stage(args)
    if stage not in STAGES:
        raise ValueError(f"--stage must be one of {STAGES}")
    if stage == "all" and (not args.fit_payloads or not args.reserved_payloads):
        raise ValueError(
            "--stage all needs --fit-payloads and --reserved-payloads (use --stage critic "
            "for the V fits and their evaluation without search payloads)"
        )
    gap = getattr(args, "gap_fit_data", None) or []
    fit = {Path(p).resolve() for p in args.fit_data}
    outside = sorted(str(p) for p in gap if Path(p).resolve() not in fit)
    if outside:
        raise ValueError(f"--gap-fit-data must list --fit-data directories only: {outside}")


def run_fit(args: argparse.Namespace, arm: ArmSpec, seed: int) -> Path:
    """initialize -> pretrain-critic (exact resume) -> A fit -> evaluate; done steps are skipped.

    ``--stage critic`` stops before the A fit and writes ``evidence-critic.json`` (V
    evidence only); a later ``--stage all`` run reuses the critic and continues at the A fit.
    """
    stage = _stage(args)
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
            "--campaign-r5",
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
            *_early_stop_flag(args),
            "--resume-state",
            work / "critic.resume.pt",
            "--resume-every",
            str(args.resume_every),
            "--progress",
            *_device_flag(args),
            *_validation_flag(args),
            "--out",
            critic,
        ]
        for directory in args.fit_data:
            command += ["--dataset", directory]
        if args.learner_state_continuations:
            command += ["--learner-state-continuations"]
        if getattr(args, "curated_scenarios", False):
            command += ["--curated-scenarios"]
        if getattr(args, "family_weights", None):
            command += ["--family-weights", args.family_weights]
        if getattr(args, "family_weight_decay", None):
            command += ["--family-weight-decay", args.family_weight_decay]
        if getattr(args, "family_weight_decay_epochs", None) is not None:
            command += ["--family-weight-decay-epochs", str(args.family_weight_decay_epochs)]
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
    if stage == "all" and not advantage.exists():
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
            *_early_stop_flag(args),
            *_device_flag(args),
        ]
        command += shlex.split(args.advantage_args or "")
        timed("advantage_fit", lambda: _run(command, log, threads))
    traces = [Path(p) for p in (getattr(args, "trajectory_trace", None) or [])]
    panel = getattr(args, "trajectory_panel", None) if stage == "all" else None
    if panel:
        trace_path = work / "critic-trace.json"
        if not trace_path.exists():
            trace_command = [
                *_python(),
                "anthropocene_rl.autonomous_v7.labor_pretrain_eval",
                "autonomous",
                "--checkpoint",
                critic,
                "--worlds",
                panel,
                "--turns",
                str(args.trajectory_turns),
                "--modes",
                "deterministic",
                "--record-critic-trace",
                "--jobs",
                "1",
                "--out",
                trace_path,
            ]
            trace_command += shlex.split(args.trajectory_autonomous_args or "")
            timed("trajectory_trace", lambda: _run(trace_command, log, threads))
        traces.append(trace_path)
    evidence_path = work / evidence_name(stage)
    if not evidence_path.exists():
        a_inputs = (
            ["--stage", "critic"]
            if stage == "critic"
            else ["--advantage", advantage, "--reserved-payloads", *args.reserved_payloads]
        )
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
            *a_inputs,
            "--calibration-data",
            *args.calibration_data,
            "--fit-data",
            *_evaluation_fit_data(args),
            "--timings",
            timings_path,
            *_device_flag(args),
            "--out",
            evidence_path,
        ]
        if getattr(args, "scenario_calibration_data", None):
            command += ["--scenario-calibration-data", args.scenario_calibration_data]
        if getattr(args, "qol_reference_scale", None):
            command += ["--qol-reference-scale", args.qol_reference_scale]
        if args.control_report:
            command += ["--control-report", args.control_report]
        if stage == "all":
            if getattr(args, "trajectory_data", None):
                command += ["--trajectory-data", *args.trajectory_data]
            for item in getattr(args, "trajectory_payload", None) or []:
                command += ["--trajectory-payload", item]
            if traces:
                command += ["--trajectory-trace", *traces]
        timed("evaluate", lambda: _run(command, log, threads))
    return evidence_path


def run_determinism(args: argparse.Namespace, arm: ArmSpec, seed: int) -> Path:
    """Two identical short critic fits and two identical A fits from the same inputs.

    ``--stage critic``: the critic probes only, in ``determinism-critic.json``; the probe
    checkpoints are reused by a later ``--stage all`` run, which adds the A probes."""
    import torch

    critic_only = _stage(args) == "critic"
    work = Path(args.out) / "determinism"
    work.mkdir(parents=True, exist_ok=True)
    out = work / ("determinism-critic.json" if critic_only else "determinism.json")
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
                "--campaign-r5",
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
                *_device_flag(args),
                "--out",
                target,
            ]
            for directory in args.fit_data:
                command += ["--dataset", directory]
            if args.learner_state_continuations:
                command += ["--learner-state-continuations"]
            if getattr(args, "curated_scenarios", False):
                command += ["--curated-scenarios"]
            if getattr(args, "family_weights", None):
                command += ["--family-weights", args.family_weights]
            if getattr(args, "family_weight_decay", None):
                command += ["--family-weight-decay", args.family_weight_decay]
            if getattr(args, "family_weight_decay_epochs", None) is not None:
                command += ["--family-weight-decay-epochs", str(args.family_weight_decay_epochs)]
            command += shlex.split(args.critic_args or "")
            _run(command, log, int(args.threads))
        critic_outputs.append(torch.load(target, map_location="cpu", weights_only=True))
    for key in ("task_critic", "cost_critics"):
        comparisons[f"critic:{key}"] = compare_state_dicts(
            critic_outputs[0][key], critic_outputs[1][key]
        )
    adv_outputs = []
    for repeat in () if critic_only else (0, 1):
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
                *_device_flag(args),
            ]
            command += shlex.split(args.advantage_args or "")
            _run(command, log, int(args.threads))
        adv_outputs.append(torch.load(target, map_location="cpu", weights_only=True))
    if not critic_only:
        comparisons["advantage:head"] = compare_state_dicts(
            adv_outputs[0]["state_dict"], adv_outputs[1]["state_dict"]
        )
    report = {
        "stage": "critic" if critic_only else "all",
        "arm": arm.name,
        "seed": seed,
        "critic_steps": int(args.determinism_steps),
        "advantage_epochs": None if critic_only else int(args.determinism_advantage_epochs),
        "threads": int(args.threads),
        "device": str(getattr(args, "device", "cpu")),
        "comparisons": comparisons,
        "tolerance": threshold(
            "determinism_tolerance_cuda"
            if str(getattr(args, "device", "cpu")).startswith("cuda")
            else "determinism_tolerance"
        ),
        "note": (
            "CUDA; same seed, data, GPU, driver and code; deterministic kernels, "
            "proposed tolerance"
            if str(getattr(args, "device", "cpu")).startswith("cuda")
            else "CPU; same seed, data, thread count and code; bitwise expected"
        ),
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
    require_calibration_not_trained(args)
    require_run_inputs(args)
    stage = _stage(args)
    if getattr(args, "qol_reference_scale", None) is not None:
        from .labor_pretrain_eval import load_qol_reference_scale

        load_qol_reference_scale(args.qol_reference_scale)  # verified before any compute
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

    def launch(batch) -> None:
        with ThreadPoolExecutor(max_workers=max(1, int(args.jobs))) as pool:
            list(pool.map(one, batch))

    gate: dict[str, Any] | None = None
    if getattr(args, "no_seed_gate", False) or len(seeds) < 2:
        launch(jobs)
    else:
        # Seed gate: every arm's first seed, then the candidate's verdict decides whether
        # the remaining seeds spend their compute.
        launch([(arm, seeds[0]) for arm in arms])
        gate = seed_gate_report(
            Path(args.out) / candidate.name / f"seed-{seeds[0]}",
            candidate,
            seeds[0],
            evidence_name(stage),
            population_tail=_population_tail(args),
        )
        rest = [(arm, seed) for seed in seeds[1:] for arm in arms]
        if gate["tripped"]:
            gate["skipped"] = [f"{arm.name}/seed-{seed}" for arm, seed in rest]
            print(
                f"[r5-acceptance] seed gate: {candidate.name} seed {seeds[0]} "
                f"{gate['reason']}; seeds {seeds[1:]} not launched",
                file=sys.stderr,
                flush=True,
            )
            jobs = [(arm, seeds[0]) for arm in arms]
        else:
            launch(rest)
        gate_path = Path(args.out) / "seed-gate.json"
        gate_path.write_text(json.dumps(gate, indent=1, sort_keys=True), encoding="utf-8")
    evidence_paths = []
    for arm, seed in jobs:
        work = Path(args.out) / arm.name / f"seed-{seed}"
        path = work / evidence_name(stage)
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
        trajectory_baseline=getattr(args, "trajectory_baseline", None),
        population_tail=_population_tail(args),
        out=Path(args.out) / ("critic-receipt.json" if stage == "critic" else "gate-receipt.json"),
        configuration={
            # "critic": V fits and their evaluation only; the A fit and gate (d) deferred.
            "stage": stage,
            "population_tail": _population_tail(args),
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
            "curated_scenarios": bool(getattr(args, "curated_scenarios", False)),
            "family_weights": getattr(args, "family_weights", None),
            "family_weight_decay": getattr(args, "family_weight_decay", None),
            "family_weight_decay_epochs": getattr(args, "family_weight_decay_epochs", None),
            "scenario_calibration_data": None
            if getattr(args, "scenario_calibration_data", None) is None
            else str(args.scenario_calibration_data),
            "calibration_data": (
                str(args.calibration_data[0])
                if len(args.calibration_data) == 1
                else [str(p) for p in args.calibration_data]
            ),
            "fit_payloads": [str(p) for p in args.fit_payloads or []],
            "reserved_payloads": [str(p) for p in args.reserved_payloads or []],
            "gap_fit_data": None
            if not getattr(args, "gap_fit_data", None)
            else [str(p) for p in args.gap_fit_data],
            "control_report": None if args.control_report is None else str(args.control_report),
            "control_report_sha256": file_sha256(args.control_report),
            # QoL gate v2: the reference-scale artifact (absent: the frozen 43.152).
            "qol_reference_scale": None
            if getattr(args, "qol_reference_scale", None) is None
            else {
                "path": str(args.qol_reference_scale),
                "sha256": file_sha256(args.qol_reference_scale),
            },
            "jobs": args.jobs,
            "threads": args.threads,
            "device": getattr(args, "device", "cpu"),
            "smoke": bool(args.smoke),
            "failures": failures,
            # Monitoring options of this harness (non-gating provenance).
            "stability_early_stop": bool(getattr(args, "stability_early_stop", False)),
            "critic_validation": bool(getattr(args, "critic_validation", False)),
            "seed_gate": gate
            if gate is not None
            else {
                "enabled": False,
                "reason": "--no-seed-gate"
                if getattr(args, "no_seed_gate", False)
                else "single seed",
            },
        },
    )
    return command_receipt(receipt_args)


def seed_gate_report(
    work: Path,
    arm: ArmSpec,
    seed: int,
    evidence_file: str = "evidence.json",
    population_tail: str = "gate",
) -> dict[str, Any]:
    """Seed gate: does the candidate's first seed end in StabilityAbort or a stability FAIL?

    Evidence: the ``*.stability-abort.json`` receipts (an abort, live early stop included) or,
    for a completed fit, the stability rows of ``fit_checks`` (the same rows the receipt
    judges). A fit that failed for another reason or left no evidence does not trip the gate
    (it HOLDs in the receipt as before)."""
    report: dict[str, Any] = {
        "enabled": True,
        "arm": arm.name,
        "seed": int(seed),
        "tripped": False,
        "reason": None,
        "evidence": None,
    }
    aborted = {}
    for label in ("critic", "advantage"):
        path = work / f"{label}.pt.stability-abort.json"
        if path.exists():
            summary = json.loads(path.read_text(encoding="utf-8"))
            aborted[label] = {
                "path": str(path),
                "mandatory_failed": summary.get("mandatory_failed"),
                "early_stop": summary.get("early_stop"),
                "rollbacks": summary.get("rollbacks"),
                "steps": summary.get("steps"),
            }
    if aborted:
        report.update(tripped=True, reason="stability-abort", evidence=aborted)
        return report
    path = work / evidence_file
    if not path.exists():
        report["reason"] = "no-evidence (fit failed for a non-stability reason or is missing)"
        return report
    evidence = json.loads(path.read_text(encoding="utf-8"))
    advisory = _advisory_checks(population_tail)
    rows = stability_checks(
        "critic", _get(evidence, "stability", "critic"), advisory=advisory
    ) + stability_checks("advantage", _get(evidence, "stability", "advantage"), advisory=advisory)
    failed = [r for r in rows if r["status"] == "FAIL"]
    if failed:
        report.update(
            tripped=True,
            reason="stability-fail",
            evidence={"path": str(path), "failed": [r["id"] for r in failed]},
        )
    else:
        report["evidence"] = {
            "path": str(path),
            "stability_statuses": {r["id"]: r["status"] for r in rows},
        }
    return report


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
    baseline_path = getattr(args, "trajectory_baseline", None)
    trajectory_baseline = (
        None
        if baseline_path is None
        else json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    )
    receipt = build_receipt(
        evidence,
        arms=parse_arms(args.arms),
        seeds=[int(s) for s in args.seeds.split(",")],
        seed_rule=args.seed_rule,
        required_seeds=int(args.required_seeds),
        determinism=determinism,
        configuration=configuration,
        attempt_id=args.attempt_id,
        trajectory_baseline=trajectory_baseline,
        population_tail=_population_tail(args),
    )
    if configuration.get("stage") == "critic":
        receipt["critic_stage"] = critic_stage_summary(receipt)
        receipt["receipt_digest"] = json_digest(
            {k: v for k, v in receipt.items() if k != "receipt_digest"}
        )
    seed_gate = configuration.get("seed_gate")
    if isinstance(seed_gate, Mapping) and seed_gate.get("tripped"):
        # The candidate's first seed failed stability: the remaining seeds were not run.
        receipt["seed_gate"] = dict(seed_gate)
        receipt["unresolved_findings"].append(
            {
                "check": "seed-gate",
                "status": "FAIL",
                "arm": seed_gate.get("arm"),
                "seed": seed_gate.get("seed"),
                "reason": seed_gate.get("reason"),
                "skipped": seed_gate.get("skipped", []),
            }
        )
        receipt["receipt_digest"] = json_digest(
            {k: v for k, v in receipt.items() if k != "receipt_digest"}
        )
    if baseline_path is not None:
        receipt["trajectory_diagnostics"]["baseline"] = {
            "path": str(baseline_path),
            "sha256": file_sha256(baseline_path),
        }
        receipt["receipt_digest"] = json_digest(
            {k: v for k, v in receipt.items() if k != "receipt_digest"}
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


# Checks a critic-stage receipt cannot judge: they read the A head, its fit or its
# stability, or the determinism of the A probe. They HOLD in the receipt and the
# critic-stage summary lists them as deferred, never as PASS.
CRITIC_STAGE_DEFERRED_PREFIXES = (
    "stability:advantage",
    "qol:a:",
    "a:",
    "gap:advantage:",
    "determinism:advantage:",
)


def critic_stage_deferred(check_id: str) -> bool:
    return any(str(check_id).startswith(p) for p in CRITIC_STAGE_DEFERRED_PREFIXES)


def critic_stage_summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """The V-only verdict of a ``--stage critic`` receipt (non-final).

    Per arm: the seed-rule status of every primary check except the A-dependent ones
    (``CRITIC_STAGE_DEFERRED_PREFIXES``) plus seed coverage; the determinism checks of the
    critic probes; the arm comparison. The machine verdict of the receipt itself stays HOLD
    (the A checks have no evidence) until ``--stage all`` writes gate-receipt.json."""
    arms: dict[str, Any] = {}
    for name, report in (receipt.get("arms") or {}).items():
        gate = report.get(GATE_ID) or {}
        rows = gate.get("checks") or []
        judged = [r for r in rows if not critic_stage_deferred(r["id"])]
        arms[name] = {
            "status": combine([r["status"] for r in judged] + [gate.get("seed_coverage", "HOLD")]),
            "seed_coverage": gate.get("seed_coverage"),
            "per_seed_status": {
                seed: combine(
                    [
                        (r["per_seed"].get(seed) or {}).get("status", "HOLD")
                        for r in judged
                    ]
                )
                for seed in sorted({s for r in judged for s in r["per_seed"]})
            },
            "not_passing": [
                {"id": r["id"], "status": r["status"], "spread": r.get("spread")}
                for r in judged
                if r["status"] != "PASS"
            ],
            "deferred": sorted(r["id"] for r in rows if critic_stage_deferred(r["id"])),
        }
    determinism = [
        r
        for r in (receipt.get("determinism") or {}).get("checks") or []
        if not critic_stage_deferred(r["id"])
    ]
    candidate = (receipt.get("verdict_basis") or {}).get("candidate_arm")
    comparison = combine(
        [
            c["status"]
            for n, c in (receipt.get("arm_comparisons") or {}).items()
            if candidate and n.startswith(f"{candidate}-vs-")
        ]
    )
    status = combine(
        [
            (arms.get(candidate) or {}).get("status", "HOLD"),
            combine([r["status"] for r in determinism]),
            comparison,
        ]
    )
    return {
        "stage": "critic",
        "final": False,
        "note": "V-only verdict; the A fit and gate (d) (same-state A ranking on the reserved "
        "worlds) are deferred until search payloads exist (--stage all)",
        "candidate_arm": candidate,
        "status": status,
        "arms": arms,
        "determinism": {
            "status": combine([r["status"] for r in determinism]),
            "checks": [r["id"] for r in determinism],
        },
        "arm_comparison": comparison,
    }


def command_evaluate(args: argparse.Namespace) -> int:
    if _stage(args) == "all" and (args.advantage is None or not args.reserved_payloads):
        raise ValueError("evaluate needs --advantage and --reserved-payloads (or --stage critic)")
    if _stage(args) == "critic" and (args.advantage is not None or args.reserved_payloads):
        raise ValueError("--stage critic evaluates the V critic only: no --advantage/payloads")
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
        trajectory_datasets=list(args.trajectory_data or []),
        trajectory_payloads=_labelled_paths(args.trajectory_payload or []),
        trajectory_traces=list(args.trajectory_trace or []),
        device=getattr(args, "device", "cpu"),
        scenario_calibration_data=args.scenario_calibration_data,
        qol_reference_scale=getattr(args, "qol_reference_scale", None),
        corrections_override={
            "tail_normalize": getattr(args, "tail_normalize", None),
            "false_terminal_policy": getattr(args, "false_terminal_policy", None),
        },
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


def command_qol_reference_scale(args: argparse.Namespace) -> int:
    """Remeasure the QoL reference scale on the FITTING batches (gate v2, user 28 Oct).

    Every world of the data must be a fitting world (90100-90153, 90160-90165): the unit
    of a held-out gate is never measured on held-out data. The artifact is versioned,
    sealed with a content digest and carries the datasets and source provenance.
    """
    from .labor_pretrain_eval import (
        qol_reference_scale_record,
        write_qol_reference_scale,
    )
    from .labor_teacher_data import load_labor_data

    paths = [Path(p) for p in args.fit_data]
    dataset = load_labor_data(paths if len(paths) > 1 else paths[0], training=False)
    worlds = sorted({int(w) for w in dataset["row_world"].reshape(-1).tolist()})
    outside = [w for w in worlds if w not in FITTING_WORLDS]
    if outside:
        raise ValueError(
            f"the QoL reference scale is measured on fitting worlds only; {outside[:5]} are not "
            "fitting worlds (90100-90153, 90160-90165)"
        )
    collection = dataset.get("collection") or {}
    record = qol_reference_scale_record(
        dataset,
        datasets=[
            {
                "paths": [str(p) for p in paths],
                "dataset_digest": collection.get("dataset_digest"),
            }
        ],
        version=int(args.version),
    )
    record["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record["source"] = source_record()
    sealed = write_qol_reference_scale(record, args.out)
    print(
        f"qol reference scale v{sealed['version']}: {sealed['value']:.4f} QoL-years "
        f"(mean {sealed['mean']:.3f}; rows {sealed['rows']}, worlds {len(sealed['worlds'])}; "
        f"G0 {sealed['previous']['value']}) digest {sealed['digest']}"
    )
    return 0


def command_td_rejudge(args: argparse.Namespace) -> int:
    """Re-judge recorded critic fits' TD refreshes under a TD rule (diagnostic only).

    Reads each stability JSONL's ``td_refresh`` records, replays them through the rule
    (``labor_stability.rejudge_td_refreshes``) and reports the live alarms (the first is
    where an early stop would have fired), the end-of-fit td_divergence status and the
    status the log's own summary recorded. It never edits a receipt.
    """
    from .labor_stability import rejudge_td_refreshes

    results = {}
    for path in args.stability_log:
        lines = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        refreshes = [r for r in lines if r.get("kind") == "td_refresh"]
        recorded = None
        for line in lines:
            if line.get("kind") in ("summary", "abort"):
                recorded = _get(line, "summary", "checks", "td_divergence", "status")
        result = rejudge_td_refreshes(refreshes, rule=args.rule)
        result["recorded_status"] = recorded
        result["path"] = str(path)
        result["sha256"] = file_sha256(path)
        results[str(path)] = result
        trajectory = (result["report"] or {}).get("trajectory") or {}
        print(
            f"{path}: {args.rule} {result['status']} (recorded {recorded}); live alarms "
            f"{[a['reason'] for a in result['live_alarms']]}; "
            + "; ".join(
                f"{key} {value.get('status')} warm-up {value.get('warmup_dropped')}"
                for key, value in trajectory.items()
            )
        )
    if args.out is not None:
        Path(args.out).write_text(
            json.dumps(results, indent=1, sort_keys=True, default=str), encoding="utf-8"
        )
    return 0


def _labelled_paths(items: Sequence[str]) -> list[tuple[str, list[Path]]]:
    """``LABEL=path[,path...]`` (repeatable) to labelled path lists in argument order."""
    out: dict[str, list[Path]] = {}
    for item in items:
        label, sep, paths = str(item).partition("=")
        if not sep or not label or not paths:
            raise ValueError(f"--trajectory-payload expects LABEL=path[,path...], got {item!r}")
        out.setdefault(label, []).extend(Path(p) for p in paths.split(",") if p)
    return list(out.items())


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


def _fmt(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def summarize(receipt: Mapping[str, Any]) -> str:
    lines = [f"{receipt['gate_id']} attempt {receipt['attempt_id']}: {receipt['machine_verdict']}"]
    if receipt.get("critic_stage"):
        stage = receipt["critic_stage"]
        lines.append(
            f"  critic stage (V only, non-final; A deferred): {stage['status']} "
            + " ".join(f"{n}={a['status']}" for n, a in stage["arms"].items())
        )
    for key, value in receipt["verdict_basis"].items():
        if key != "secondary_gate_note":
            lines.append(f"  {key}: {value}")
    for name, report in receipt["arms"].items():
        gate = report[GATE_ID]
        lines.append(
            f"  arm {name}: {gate['status']} rule={gate['rule']} seeds={gate['seeds']} "
            f"every-seed={gate['passes_in_every_seed']} per-seed={gate['per_seed_status']}"
        )
        for seed, diagnostics in (report.get("diagnostics") or {}).items():
            for stratum, m in (diagnostics.get("qol") or {}).items():
                if stratum != "all":
                    continue
                lines.append(
                    f"    seed {seed} diagnostics (non-gating, QoL ref scales; constant "
                    f"reference in brackets): n-step nMAE={_fmt(m['nstep_nmae'])} "
                    f"[{_fmt(m['nstep_nmae_constant'])}] bias={_fmt(m['nstep_bias'])} "
                    f"[{_fmt(m['nstep_bias_constant'])}] | MC nMAE={_fmt(m['mc_nmae'])} "
                    f"[{_fmt(m['mc_nmae_constant'])}] bias={_fmt(m['mc_bias'])} "
                    f"[{_fmt(m['mc_bias_constant'])}]"
                )
    comparison = (receipt.get("trajectory_diagnostics") or {}).get("comparison")
    if comparison is not None:
        lines.append(
            f"  trajectory diagnostics (non-gating): {comparison['flagged_count']} T-metric(s) "
            "worse than their seed spread against the baseline"
        )
    for finding in receipt["unresolved_findings"][:30]:
        lines.append(f"  - {finding}")
    return "\n".join(lines)


def argument_parser() -> argparse.ArgumentParser:
    from .labor_device import add_device_argument

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    evaluate = sub.add_parser("evaluate", help="one fit's evidence JSON")
    evaluate.add_argument(
        "--tail-normalize",
        choices=("none", "world-factor"),
        default=None,
        help="apply this tail normalisation to the calibration and fit data whatever the "
        "checkpoint recorded (default: the recorded correction)",
    )
    evaluate.add_argument(
        "--false-terminal-policy",
        choices=("none", "mask"),
        default=None,
        help="apply this false-terminal policy to the calibration and fit data whatever the "
        "checkpoint recorded (default: the recorded correction)",
    )
    evaluate.add_argument("--arm", required=True)
    evaluate.add_argument("--seed", type=int, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument(
        "--stage",
        choices=STAGES,
        default="all",
        help="critic: V evidence only (no --advantage/--reserved-payloads; A checks HOLD)",
    )
    evaluate.add_argument("--advantage", type=Path, default=None)
    evaluate.add_argument(
        "--calibration-data",
        type=Path,
        nargs="+",
        required=True,
        help="calibration dataset(s); several batches are read as one (world order)",
    )
    evaluate.add_argument(
        "--fit-data",
        type=Path,
        nargs="+",
        default=None,
        help="teacher fitting dataset(s) for the generalisation gap; several are read as one",
    )
    evaluate.add_argument("--reserved-payloads", type=Path, nargs="+", default=None)
    evaluate.add_argument(
        "--control-report",
        type=Path,
        default=None,
        help="critic3 critic-calibration JSON on calibration worlds 90180-90199",
    )
    evaluate.add_argument("--timings", type=Path, default=None)
    evaluate.add_argument(
        "--scenario-calibration-data",
        type=Path,
        default=None,
        help="labor_scenarios calibration scenario set (report-only per-family calibration)",
    )
    evaluate.add_argument(
        "--qol-reference-scale",
        type=Path,
        default=None,
        help="QoL reference-scale artifact (qol-reference-scale): the unit of the QoL "
        "calibration gate and the TD residual; absent, the frozen 43.152",
    )
    evaluate.add_argument("--batch-size", type=int, default=256)
    add_device_argument(evaluate)
    _trajectory_arguments(evaluate)
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
    receipt.add_argument(
        "--trajectory-baseline",
        type=Path,
        default=None,
        help="earlier receipt: flag T-metrics worse than their seed spread (non-gating)",
    )
    _population_tail_argument(receipt)
    receipt.add_argument("--out", type=Path, required=True)
    receipt.set_defaults(handler=command_receipt)

    scale = sub.add_parser(
        "qol-reference-scale",
        help="remeasure the QoL reference scale on the fitting batches (versioned artifact)",
    )
    scale.add_argument(
        "--fit-data",
        type=Path,
        nargs="+",
        required=True,
        help="the r5 fitting batches (fitting worlds only), read as one dataset",
    )
    scale.add_argument("--version", type=int, default=1)
    scale.add_argument("--out", type=Path, required=True)
    scale.set_defaults(handler=command_qol_reference_scale)

    rejudge = sub.add_parser(
        "td-rejudge", help="re-judge recorded TD refresh series under a TD rule (diagnostic)"
    )
    rejudge.add_argument("--stability-log", type=Path, nargs="+", required=True)
    rejudge.add_argument(
        "--rule", choices=("v1-diminishing", "v2-floor-or-declining"),
        default="v2-floor-or-declining",
    )
    rejudge.add_argument("--out", type=Path, default=None)
    rejudge.set_defaults(handler=command_td_rejudge)

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
        help="r5 teacher fitting dataset(s), e.g. the collection batches fit8-b01..b08 (read "
        "as one teacher dataset: world order, equal mass per world), plus optional "
        "continuation or curated scenario datasets",
    )
    run.add_argument("--learner-state-continuations", action="store_true", default=False)
    run.add_argument(
        "--family-weights",
        default=None,
        help="pretrain-critic --family-weights for every critic fit (labor_scenarios; "
        "'default' = the proposed weights)",
    )
    run.add_argument("--family-weight-decay", choices=("none", "linear", "cosine"), default=None)
    run.add_argument("--family-weight-decay-epochs", type=float, default=None)
    run.add_argument(
        "--curated-scenarios",
        action="store_true",
        default=False,
        help="admit curated scenario fitting sets listed in --fit-data (labor_scenarios)",
    )
    run.add_argument(
        "--calibration-data",
        type=Path,
        nargs="+",
        required=True,
        help="calibration dataset(s), e.g. cal8-b01..b03 (read as one)",
    )
    run.add_argument(
        "--scenario-calibration-data",
        type=Path,
        default=None,
        help="labor_scenarios calibration scenario set: report-only per-family calibration "
        "(fitting scenario sets join --fit-data with --critic-args --curated-scenarios)",
    )
    run.add_argument(
        "--fit-payloads",
        type=Path,
        nargs="+",
        default=None,
        help="search candidate payloads from fitting worlds (A pairs); required by --stage all",
    )
    run.add_argument(
        "--reserved-payloads",
        type=Path,
        nargs="+",
        default=None,
        help="held-out-test candidate payloads of reserved worlds 90154-90159; required by "
        "--stage all",
    )
    run.add_argument(
        "--stage",
        choices=STAGES,
        default="all",
        help="all: V fit, A fit, evaluate, gate-receipt.json. critic: V fits, their per-epoch "
        "validation, the critic determinism probe and the V evaluation (relative ordering, "
        "TD(0), calibration) without search payloads -> evidence-critic.json and "
        "critic-receipt.json (A checks HOLD as deferred); a later --stage all in the same "
        "--out continues at the A fit",
    )
    run.add_argument(
        "--gap-fit-data",
        type=Path,
        nargs="+",
        default=None,
        help="fitting directories (a subset of --fit-data) the evaluation reads for the "
        "generalisation gap; default every --fit-data directory (bounds evaluate's host "
        "memory on small hosts)",
    )
    run.add_argument("--control-report", type=Path, default=None)
    run.add_argument(
        "--qol-reference-scale",
        type=Path,
        default=None,
        help="QoL reference-scale artifact (qol-reference-scale) passed to every evaluate; "
        "absent, the frozen 43.152",
    )
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
    run.add_argument(
        "--critic-args",
        default="",
        help="extra pretrain-critic arguments (e.g. --qol-loss-weight 30 --loss-balancing "
        "gradnorm --grad-clip per-encoder)",
    )
    run.add_argument("--advantage-args", default="", help="extra advantage fit arguments")
    run.add_argument("--batch-size", type=int, default=256)
    run.add_argument("--lr", type=float, default=3e-4)
    run.add_argument("--max-grad-norm", type=float, default=1.0)
    run.add_argument("--advantage-epochs", type=int, default=50)
    run.add_argument("--stability-every", type=int, default=25)
    run.add_argument(
        "--no-stability-early-stop",
        dest="stability_early_stop",
        action="store_false",
        default=True,
        help="judge gradient dominance, TD divergence and total rollbacks only at the end of "
        "each fit (by default a fit stops the moment one of them trips: same thresholds)",
    )
    run.add_argument(
        "--no-critic-validation",
        dest="critic_validation",
        action="store_false",
        default=True,
        help="skip the V critic's per-epoch held-out validation on --calibration-data "
        "(monitoring only; the fit is bitwise unchanged either way)",
    )
    run.add_argument(
        "--no-seed-gate",
        action="store_true",
        default=False,
        help="launch every seed at once (by default the first seed of every arm runs first and "
        "a candidate StabilityAbort or stability FAIL stops the remaining seeds)",
    )
    run.add_argument("--resume-every", type=int, default=100)
    run.add_argument("--determinism-steps", type=int, default=50)
    run.add_argument("--determinism-advantage-epochs", type=int, default=2)
    run.add_argument("--jobs", type=int, default=4)
    run.add_argument("--threads", type=int, default=14)
    run.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="device for pretrain-critic, the A fit, evaluate and the determinism probes "
        "(stored data only); initialize and the T4 trajectory recorder stay on CPU",
    )
    run.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help="local mechanics smoke: admit truncated-zero targets; the receipt is forced to HOLD",
    )
    run.add_argument("--attempt-id", required=True)
    _population_tail_argument(run)
    _trajectory_arguments(run)
    run.add_argument("--trajectory-baseline", type=Path, default=None)
    run.add_argument(
        "--trajectory-panel",
        default=None,
        help="development worlds for the T4 recorder (labor_pretrain_eval autonomous "
        "--record-critic-trace, deterministic mode) run per fit; off when absent",
    )
    run.add_argument("--trajectory-turns", type=int, default=25)
    run.add_argument("--trajectory-autonomous-args", default="")
    run.add_argument("--out", type=Path, required=True)
    run.set_defaults(handler=command_run)
    return parser


def _population_tail_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--population-tail",
        choices=POPULATION_TAIL_POLICIES,
        default=threshold("population_tail_policy_r5"),
        help="advisory (r5 default, user 28 Oct): the population-tail stability check is "
        "reported under the fit's advisories and escalated, but never fails stability or "
        "trips the seed gate; gate: the 26 Oct mandatory rule",
    )


def _trajectory_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--trajectory-data",
        type=Path,
        nargs="*",
        default=[],
        help="T3: extra critic datasets (teacher, DAgger, learner-state continuations)",
    )
    parser.add_argument(
        "--trajectory-payload",
        action="append",
        default=[],
        help="T3: LABEL=payload[,payload...] candidate sets (e.g. round-1=...); repeatable",
    )
    parser.add_argument(
        "--trajectory-trace",
        type=Path,
        nargs="*",
        default=[],
        help="T4: autonomous reports recorded with --record-critic-trace",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    return int(args.handler(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
