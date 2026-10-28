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
#   FIT_DATA          r5 teacher fitting dataset dir(s) (90100-90153, 90160-90165): one dir
#                     or the collection batches, e.g. "$I/fit8-b01 $I/fit8-b02 ..." or
#                     "$I/fit8-b01:$I/fit8-b02:...". Several teacher batches are read as ONE
#                     teacher dataset (shards in world order, equal mass per world; a world
#                     in two batches, a repeated batch, a normalizer/tail/turns mismatch or
#                     a reserved/calibration world is refused). Learner-state continuation
#                     (CONTINUATIONS=1) or curated scenario (SCENARIOS=1) sets may follow;
#                     they keep their own source mass.
#   CAL_DATA          r5 calibration dataset dir(s) (90180-90199), e.g. the batches
#                     cal8-b01..b03; same list syntax, read as one dataset
#   FIT_PAYLOADS      search candidate payload dir(s) of fitting worlds (A pairs); not used
#                     (and not required) with STAGE=critic
#   RESERVED_PAYLOADS held-out-test payload dir(s) of reserved worlds 90154-90159; not used
#                     (and not required) with STAGE=critic
# Every list variable (FIT_DATA, CAL_DATA, FIT_PAYLOADS, RESERVED_PAYLOADS) is space- or
# colon-separated; paths must not contain spaces or colons.
#   CONTROL_REPORT    critic3 critic-calibration JSON on 90180-90199 (other-head control)
#   OUT               output directory (gate-receipt.json lands here)
#   ATTEMPT_ID        receipt attempt id (new id for every corrected attempt)
# Optional:
#   STAGE=all         critic: V fits (seed gate, per-epoch validation, early stop), the critic
#                     determinism probe and the V evaluation (relative ordering, TD(0),
#                     calibration) without search payloads -> <arm>/seed-<s>/evidence-critic.json
#                     and critic-receipt.json (A checks HOLD as deferred). A later STAGE=all run
#                     with the same OUT continues at the A fit (critics are not refitted).
#   GAP_FIT_DATA      fitting dirs (a subset of FIT_DATA) evaluate reads for the generalisation
#                     gap; default all of FIT_DATA (bounds evaluate's host memory)
#   POPULATION_TAIL=advisory  run --population-tail: advisory (r5 default, user 28 Oct) reports
#                     the population-tail stability check without gating it; gate = 26 Oct rule
#   CONTROL_REPORT absent: the other heads are compared with the run's own seed-matched plain
#                     control arm (user 28 Oct)
#   LR=3e-4  INIT_ARGS="..."  CRITIC_ARGS="..."  ADVANTAGE_ARGS="..."  (run --lr/--init-args/
#                     --critic-args/--advantage-args, quoted as one argument each)
#   SEEDS=0,1,2,3,4  ARMS=privileged-e4:privileged:4:candidate,control-e4:plain:4:control
#   JOBS=4 THREADS=14 (4 fits x 14 threads on the 60-core VM)  SEED_RULE=mean-with-floor
#   ADVANTAGE_EPOCHS=50  RESUME_EVERY=100  DETERMINISM_STEPS=50  PYTHON=python3
#   CONTINUATIONS=0  EXTRA="..." (passed through to labor_r5_acceptance run)
#   DEVICE=cpu (cuda on the GPU fitting VM: critic, A fit, evaluate and determinism probes
#   run on the GPU; JOBS fits share it, THREADS is then the host threads per fit;
#   TORCH_DISABLE_NATIVE_JIT=1 is exported below for every fit)
#   SCENARIOS=0       1: FIT_DATA also lists curated scenario fitting sets (labor_scenarios)
#   SCENARIO_CAL_DATA calibration scenario set (report-only per-family calibration)
#   QOL_REFERENCE_SCALE  QoL reference-scale artifact (labor_r5_acceptance qol-reference-scale on
#                     the fitting batches): the unit of the QoL gate; absent, the frozen 43.152
#   FAMILY_WEIGHTS    pretrain-critic --family-weights (e.g. default; proposed values)
#   EARLY_STOP=1      live stability early stop in every V and A fit (gradient_dominance or
#                     td_divergence alarm, or >5 rollbacks: same thresholds as the end-of-run
#                     checks); 0 judges them only at the end of each fit
#   CRITIC_VALIDATION=1  per-epoch V validation on CAL_DATA (monitoring only: non-gating
#                     validation_divergence advisory, fits bitwise unchanged); 0 skips it
#   SEED_GATE=1       run the first seed of every arm first; a candidate StabilityAbort or
#                     stability FAIL stops the remaining seeds (seed-gate.json); 0 launches all
#   FAMILY_WEIGHT_DECAY (none|linear|cosine; cosine by default with weights)  FAMILY_WEIGHT_DECAY_EPOCHS
#
# Never point any input at sealed worlds 3000-3059; the reserved worlds 90154-90159 are
# only ever read as RESERVED_PAYLOADS (held-out tests). Every loader refuses violations.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${NORMALIZER:?}" "${PRIV_NORMALIZER:?}" "${FIT_DATA:?}" "${CAL_DATA:?}"
: "${OUT:?}" "${ATTEMPT_ID:?}"
STAGE="${STAGE:-all}"
if [ "$STAGE" = "all" ]; then : "${FIT_PAYLOADS:?}" "${RESERVED_PAYLOADS:?}"; fi
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
EARLY_STOP="${EARLY_STOP:-1}"
CRITIC_VALIDATION="${CRITIC_VALIDATION:-1}"
SEED_GATE="${SEED_GATE:-1}"

cores="$(nproc 2>/dev/null || echo 0)"
if [ "$cores" -gt 0 ] && [ $((JOBS * THREADS)) -gt "$cores" ]; then
  echo "[r5-acceptance] JOBS*THREADS=$((JOBS * THREADS)) exceeds $cores cores" >&2
  exit 2
fi

mkdir -p "$OUT"
# Space- or colon-separated path lists -> arrays.
split_list() { local value="${1//:/ }"; read -r -a "$2" <<<"$value"; }
split_list "$FIT_DATA" fit_data
split_list "$CAL_DATA" cal_data
fit_payloads=(); reserved_payloads=(); gap_fit_data=()
if [ -n "${FIT_PAYLOADS:-}" ]; then split_list "$FIT_PAYLOADS" fit_payloads; fi
if [ -n "${RESERVED_PAYLOADS:-}" ]; then split_list "$RESERVED_PAYLOADS" reserved_payloads; fi
if [ -n "${GAP_FIT_DATA:-}" ]; then split_list "$GAP_FIT_DATA" gap_fit_data; fi
# CUDA fits: torch 2.14's torch._native Triton kernels need a C compiler the GPU VM lacks;
# this switch (read once at `import torch`, inherited by every fit subprocess) keeps them
# unregistered. Harmless on CPU. labor_device.configure refuses CUDA without it.
export TORCH_DISABLE_NATIVE_JIT=1
export PYTHONPATH="$here/src${PYTHONPATH:+:$PYTHONPATH}"
args=(
  --normalizer-checkpoint "$NORMALIZER"
  --privileged-normalizer "$PRIV_NORMALIZER"
  --fit-data "${fit_data[@]}"
  --calibration-data "${cal_data[@]}"
  --stage "$STAGE"
  --arms "$ARMS" --seeds "$SEEDS" --seed-rule "$SEED_RULE"
  --advantage-epochs "$ADVANTAGE_EPOCHS" --resume-every "$RESUME_EVERY"
  --determinism-steps "$DETERMINISM_STEPS"
  --jobs "$JOBS" --threads "$THREADS" --device "$DEVICE"
  --attempt-id "$ATTEMPT_ID" --out "$OUT"
)
if [ "${#fit_payloads[@]}" -gt 0 ]; then args+=(--fit-payloads "${fit_payloads[@]}"); fi
if [ "${#reserved_payloads[@]}" -gt 0 ]; then args+=(--reserved-payloads "${reserved_payloads[@]}"); fi
if [ "${#gap_fit_data[@]}" -gt 0 ]; then args+=(--gap-fit-data "${gap_fit_data[@]}"); fi
if [ -n "${LR:-}" ]; then args+=(--lr "$LR"); fi
if [ -n "${POPULATION_TAIL:-}" ]; then args+=(--population-tail "$POPULATION_TAIL"); fi
if [ -n "${INIT_ARGS:-}" ]; then args+=(--init-args "$INIT_ARGS"); fi
if [ -n "${CRITIC_ARGS:-}" ]; then args+=(--critic-args "$CRITIC_ARGS"); fi
if [ -n "${ADVANTAGE_ARGS:-}" ]; then args+=(--advantage-args "$ADVANTAGE_ARGS"); fi
if [ -n "${CONTROL_REPORT:-}" ]; then args+=(--control-report "$CONTROL_REPORT"); fi
if [ "$CONTINUATIONS" = "1" ]; then args+=(--learner-state-continuations); fi
if [ "$SCENARIOS" = "1" ]; then args+=(--curated-scenarios); fi
if [ "$EARLY_STOP" = "0" ]; then args+=(--no-stability-early-stop); fi
if [ "$CRITIC_VALIDATION" = "0" ]; then args+=(--no-critic-validation); fi
if [ "$SEED_GATE" = "0" ]; then args+=(--no-seed-gate); fi
if [ -n "${FAMILY_WEIGHTS:-}" ]; then args+=(--family-weights "$FAMILY_WEIGHTS"); fi
if [ -n "${FAMILY_WEIGHT_DECAY:-}" ]; then args+=(--family-weight-decay "$FAMILY_WEIGHT_DECAY"); fi
if [ -n "${FAMILY_WEIGHT_DECAY_EPOCHS:-}" ]; then args+=(--family-weight-decay-epochs "$FAMILY_WEIGHT_DECAY_EPOCHS"); fi
if [ -n "${SCENARIO_CAL_DATA:-}" ]; then args+=(--scenario-calibration-data "$SCENARIO_CAL_DATA"); fi
if [ -n "${QOL_REFERENCE_SCALE:-}" ]; then args+=(--qol-reference-scale "$QOL_REFERENCE_SCALE"); fi
# shellcheck disable=SC2206
if [ -n "${EXTRA:-}" ]; then args+=($EXTRA); fi

echo "[r5-acceptance] $(date -u +%FT%TZ) start stage=$STAGE attempt=$ATTEMPT_ID arms=$ARMS seeds=$SEEDS jobs=$JOBS threads=$THREADS device=$DEVICE early_stop=$EARLY_STOP critic_validation=$CRITIC_VALIDATION seed_gate=$SEED_GATE"
"$PYTHON" -m anthropocene_rl.autonomous_v7.labor_r5_acceptance run "${args[@]}" 2>&1 | tee -a "$OUT/driver.log"
status=${PIPESTATUS[0]}
receipt="$OUT/gate-receipt.json"; [ "$STAGE" = "critic" ] && receipt="$OUT/critic-receipt.json"
echo "[r5-acceptance] $(date -u +%FT%TZ) done status=$status receipt=$receipt"
exit "$status"
