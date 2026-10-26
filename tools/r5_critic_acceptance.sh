#!/usr/bin/env bash
# r5 critic acceptance on the VM: arm x seed critic (V) and advantage (A) fits with the
# enforced stability monitor, the existing evaluators, the determinism probe and the
# machine gate receipt (labor_r5_acceptance). Resumable: rerun the same command after a
# preemption; finished stages are skipped and a running critic fit continues bit-for-bit
# from its last resume state (critic.resume.pt, written every RESUME_EVERY steps).
#
# Required environment (absolute paths on the VM):
#   NORMALIZER        actor normalizer checkpoint refitted on r5 fitting worlds
#   PRIV_NORMALIZER   privileged normalizer v2 fitted on r5 fitting worlds
#   FIT_DATA          r5 teacher fitting dataset dir (90100-90153, 90160-90165); more dirs
#                     may follow, space separated (learner-state continuations, with
#                     CONTINUATIONS=1)
#   CAL_DATA          r5 calibration dataset dir (90180-90199)
#   FIT_PAYLOADS      search candidate payload dir(s) of fitting worlds (A pairs)
#   RESERVED_PAYLOADS held-out-test payload dir(s) of reserved worlds 90154-90159
#   CONTROL_REPORT    critic3 critic-calibration JSON on 90180-90199 (other-head control)
#   OUT               output directory (gate-receipt.json lands here)
#   ATTEMPT_ID        receipt attempt id (new id for every corrected attempt)
# Optional:
#   SEEDS=0,1,2,3,4  ARMS=privileged-e4:privileged:4:candidate,control-e4:plain:4:control
#   JOBS=4 THREADS=14 (4 fits x 14 threads on the 60-core VM)  SEED_RULE=mean-with-floor
#   ADVANTAGE_EPOCHS=50  RESUME_EVERY=100  DETERMINISM_STEPS=50  PYTHON=python3
#   CONTINUATIONS=0  EXTRA="..." (passed through to labor_r5_acceptance run)
#   DEVICE=cpu (cuda on the GPU fitting VM: critic, A fit, evaluate and determinism probes
#   run on the GPU; JOBS fits share it, THREADS is then the host threads per fit)
#   SCENARIOS=0       1: FIT_DATA also lists curated scenario fitting sets (labor_scenarios)
#   SCENARIO_CAL_DATA calibration scenario set (report-only per-family calibration)
#
# Never point any input at sealed worlds 3000-3059; the reserved worlds 90154-90159 are
# only ever read as RESERVED_PAYLOADS (held-out tests). Every loader refuses violations.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${NORMALIZER:?}" "${PRIV_NORMALIZER:?}" "${FIT_DATA:?}" "${CAL_DATA:?}"
: "${FIT_PAYLOADS:?}" "${RESERVED_PAYLOADS:?}" "${OUT:?}" "${ATTEMPT_ID:?}"
PYTHON="${PYTHON:-python3}"
SEEDS="${SEEDS:-0,1,2,3,4}"
ARMS="${ARMS:-privileged-e4:privileged:4:candidate,control-e4:plain:4:control}"
JOBS="${JOBS:-4}"
THREADS="${THREADS:-14}"
SEED_RULE="${SEED_RULE:-mean-with-floor}"
ADVANTAGE_EPOCHS="${ADVANTAGE_EPOCHS:-50}"
RESUME_EVERY="${RESUME_EVERY:-100}"
DETERMINISM_STEPS="${DETERMINISM_STEPS:-50}"
CONTINUATIONS="${CONTINUATIONS:-0}"
DEVICE="${DEVICE:-cpu}"
SCENARIOS="${SCENARIOS:-0}"

cores="$(nproc 2>/dev/null || echo 0)"
if [ "$cores" -gt 0 ] && [ $((JOBS * THREADS)) -gt "$cores" ]; then
  echo "[r5-acceptance] JOBS*THREADS=$((JOBS * THREADS)) exceeds $cores cores" >&2
  exit 2
fi

mkdir -p "$OUT"
export PYTHONPATH="$here/src${PYTHONPATH:+:$PYTHONPATH}"
args=(
  --normalizer-checkpoint "$NORMALIZER"
  --privileged-normalizer "$PRIV_NORMALIZER"
  # shellcheck disable=SC2206
  --fit-data $FIT_DATA
  --calibration-data "$CAL_DATA"
  # shellcheck disable=SC2206
  --fit-payloads $FIT_PAYLOADS
  # shellcheck disable=SC2206
  --reserved-payloads $RESERVED_PAYLOADS
  --arms "$ARMS" --seeds "$SEEDS" --seed-rule "$SEED_RULE"
  --advantage-epochs "$ADVANTAGE_EPOCHS" --resume-every "$RESUME_EVERY"
  --determinism-steps "$DETERMINISM_STEPS"
  --jobs "$JOBS" --threads "$THREADS" --device "$DEVICE"
  --attempt-id "$ATTEMPT_ID" --out "$OUT"
)
if [ -n "${CONTROL_REPORT:-}" ]; then args+=(--control-report "$CONTROL_REPORT"); fi
if [ "$CONTINUATIONS" = "1" ]; then args+=(--learner-state-continuations); fi
if [ "$SCENARIOS" = "1" ]; then args+=(--curated-scenarios); fi
if [ -n "${SCENARIO_CAL_DATA:-}" ]; then args+=(--scenario-calibration-data "$SCENARIO_CAL_DATA"); fi
# shellcheck disable=SC2206
if [ -n "${EXTRA:-}" ]; then args+=($EXTRA); fi

echo "[r5-acceptance] $(date -u +%FT%TZ) start attempt=$ATTEMPT_ID arms=$ARMS seeds=$SEEDS jobs=$JOBS threads=$THREADS device=$DEVICE"
"$PYTHON" -m anthropocene_rl.autonomous_v7.labor_r5_acceptance run "${args[@]}" 2>&1 | tee -a "$OUT/driver.log"
status=${PIPESTATUS[0]}
echo "[r5-acceptance] $(date -u +%FT%TZ) done status=$status receipt=$OUT/gate-receipt.json"
exit "$status"
