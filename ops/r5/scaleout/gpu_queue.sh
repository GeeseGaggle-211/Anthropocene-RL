#!/usr/bin/env bash
# r5 scale-out, GPU VM (anthropocene-gpu-fit-l4: 1 L4, 16 vCPU, 62 GB host RAM).
#
# Order (each step skipped when its output exists; rerun the same command after any stop):
#   1. preflight (GPU, venv, code, bridge, disk, RAM) and staging of the r5 inputs from GCS;
#   2. the QoL reference scale remeasured on all 8 fitting batches (r5-scale-fit8.json);
#   3. optional (BC=1, default): the actor BC pilot (actor_bc_pilot.sh, DEVICE=cuda), uploaded
#      to $SCALEOUT_GCS/actor-pilot/ so the CPU VM's panel can start;
#   4. the critic V scale-out: tools/r5_critic_acceptance.sh STAGE=critic with the approved
#      G1 configuration, 2 arms x 5 seeds, seed gate, JOBS=1 (host RAM, see README);
#   5. upload of everything to $SCALEOUT_GCS/{critic,actor-pilot,gpu-logs}/, then shutdown.
# The EXIT trap uploads the logs and shuts the VM down on success AND on failure
# (NO_SHUTDOWN=1 to keep it up while debugging).
#
# Usage on the VM (detached, survives the SSH session):
#   CODE=$HOME/r5gpu/code-<sha>/rl setsid nohup bash $CODE/ops/r5/scaleout/gpu_queue.sh \
#     > $HOME/r5gpu/scaleout/queue.out 2>&1 < /dev/null &
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "$here/lib.sh"
STAGE_NAME=gpu-queue

R="${R5GPU:-$HOME/r5gpu}"
CODE="${CODE:-$(cd "$here/../../.." && pwd)}"
PY="${PY:-$R/venv/bin/python}"
DATA="${DATA:-$R/pilot}"
WORK="${WORK:-$R/scaleout}"
BC="${BC:-1}"
export ANTHROPOCENE_DOTNET="${ANTHROPOCENE_DOTNET:-$R/bridge/dotnet/dotnet}"
export ANTHROPOCENE_MODEL9_BRIDGE="${ANTHROPOCENE_MODEL9_BRIDGE:-$R/bridge/bridge/Anthropocene.Model9Bridge.dll}"
export TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 PYTHONPATH="$CODE/src"

mkdir -p "$WORK" "$DATA"
LOG_FILE="$WORK/queue.log"
memwatch_pid=""
uploader_pid=""

finish() {
  local status=$?
  trap - EXIT
  [ -n "$uploader_pid" ] && kill "$uploader_pid" 2> /dev/null || true
  [ -n "$memwatch_pid" ] && kill "$memwatch_pid" 2> /dev/null || true
  if [ "$status" -eq 0 ]; then log "queue finished"; else log "queue FAILED status=$status"; fi
  [ -f "$WORK/memwatch.csv" ] && log "memory: $(memwatch_peak "$WORK/memwatch.csv")"
  echo "$status $(ts)" > "$WORK/EXIT_STATUS"
  [ -d "$WORK/critic" ] && gcs_sync "$WORK/critic" "$SCALEOUT_GCS/critic" '.*\.resume\.pt$'
  for f in queue.log queue.out memwatch.csv EXIT_STATUS r5-scale-fit8.json r5-scale-fit8.log; do
    [ -f "$WORK/$f" ] && gcs_put "$WORK/$f" "$SCALEOUT_GCS/gpu-logs/$f"
  done
  shutdown_vm
  exit "$status"
}
trap finish EXIT
trap 'log "signal received"; exit 143' TERM INT HUP

# 1. Preflight -------------------------------------------------------------------------
log "start code=$CODE bc=$BC"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader >> "$LOG_FILE" 2>&1 \
  || die "nvidia-smi failed"
require_file "$PY" "venv"
require_file "$CODE/tools/r5_critic_acceptance.sh" "learner checkout"
require_file "$ANTHROPOCENE_DOTNET" "portable dotnet (initialize probes the bridge)"
require_file "$ANTHROPOCENE_MODEL9_BRIDGE" "portable bridge"
help_text=$("$PY" -m anthropocene_rl.autonomous_v7.labor_r5_acceptance run --help < /dev/null 2>&1 || true)
case "$help_text" in
  *--population-tail*) ;;
  *) die "learner checkout lacks run --population-tail (needs ops/r5-scaleout 839d197 or later)" ;;
esac
"$PY" - <<'PY' >> "$LOG_FILE" 2>&1 || die "torch CUDA unavailable"
import torch
assert torch.cuda.is_available()
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
PY
free_gb=$(disk_free_gb "$R")
[ "$free_gb" -ge "${MIN_DISK_GB:-40}" ] || die "only ${free_gb} GB free under $R (need ${MIN_DISK_GB:-40})"
log "disk free ${free_gb} GB, MemAvailable $(mem_available_gb) GB"
memwatch_pid=$(memwatch_start "$WORK/memwatch.csv" 60)

# Inputs: the 8 fitting and 3 calibration batches plus both normalizers.
for b in $FIT_BATCHES_ALL $CAL_BATCHES_ALL; do
  log "stage $b"
  gcs_fetch_dir "$GCS_ROOT/pilot/$b" "$DATA/$b"
done
NORMALIZER="${NORMALIZER:-$DATA/actor-normalizer.pt}"
PRIV_NORMALIZER="${PRIV_NORMALIZER:-$DATA/privileged-normalizer-r5.pt}"
[ -f "$NORMALIZER" ] || timeout 600 gcloud storage cp "$GCS_ROOT/pilot/actor-normalizer.pt" "$NORMALIZER" < /dev/null
[ -f "$PRIV_NORMALIZER" ] || timeout 600 gcloud storage cp "$GCS_ROOT/post-collection/privileged-normalizer.pt" "$PRIV_NORMALIZER" < /dev/null
[ "$(sha256_of "$NORMALIZER")" = "$ACTOR_NORMALIZER_SHA256" ] || die "actor normalizer sha256 mismatch"
log "privileged normalizer sha256 $(sha256_of "$PRIV_NORMALIZER" | cut -c1-16)"

fit_dirs=$(join_paths "$DATA" $FIT_BATCHES_ALL)
cal_dirs=$(join_paths "$DATA" $CAL_BATCHES_ALL)

# Background uploader: the critic outputs every UPLOAD_EVERY seconds (resume states excluded).
(
  while sleep "${UPLOAD_EVERY:-1800}"; do
    [ -d "$WORK/critic" ] && gcs_sync "$WORK/critic" "$SCALEOUT_GCS/critic" '.*\.resume\.pt$'
    gcs_put "$WORK/queue.log" "$SCALEOUT_GCS/gpu-logs/queue.log"
    gcs_put "$WORK/memwatch.csv" "$SCALEOUT_GCS/gpu-logs/memwatch.csv"
  done
) > /dev/null 2>&1 < /dev/null &
uploader_pid=$!

# 2. QoL reference scale on all 8 fitting batches (replaces out-v2's 5-batch artifact).
SCALE="$WORK/r5-scale-fit8.json"
if [ ! -f "$SCALE" ]; then
  log "qol-reference-scale on $FIT_BATCHES_ALL"
  # shellcheck disable=SC2086
  timeout 7200 "$PY" -m anthropocene_rl.autonomous_v7.labor_r5_acceptance qol-reference-scale \
    --fit-data $fit_dirs --out "$SCALE" > "$WORK/r5-scale-fit8.log" 2>&1 < /dev/null \
    || die "qol-reference-scale (see r5-scale-fit8.log)"
  log "$(tail -n 1 "$WORK/r5-scale-fit8.log")"
  gcs_put "$SCALE" "$SCALEOUT_GCS/r5-scale-fit8.json"
fi

# 3. Actor BC pilot on the GPU (before the critic: the two cannot share 62 GB of host RAM).
if [ "$BC" = "1" ]; then
  log "actor BC pilot"
  CODE="$CODE" PY="$PY" DATA="$DATA" NORMALIZER="$NORMALIZER" PRIV_NORMALIZER="$PRIV_NORMALIZER" \
    OUTD="$WORK/actor-pilot" DEVICE=cuda THREADS="${BC_THREADS:-12}" \
    bash "$here/actor_bc_pilot.sh" || die "actor BC pilot"
fi

# 4. Critic V scale-out: the approved G1 configuration, critic stage (no search payloads).
CRITIC_OUT="$WORK/critic"
if [ ! -f "$CRITIC_OUT/critic-receipt.json" ]; then
  log "critic V scale-out (stage critic) -> $CRITIC_OUT"
  NORMALIZER="$NORMALIZER" PRIV_NORMALIZER="$PRIV_NORMALIZER" \
  FIT_DATA="$fit_dirs" CAL_DATA="$cal_dirs" OUT="$CRITIC_OUT" \
  ATTEMPT_ID="${ATTEMPT_ID:-r5-scaleout-critic-g1-a1}" \
  STAGE=critic PYTHON="$PY" DEVICE=cuda \
  JOBS="${JOBS:-1}" THREADS="${THREADS:-12}" \
  SEEDS="${SEEDS:-0,1,2,3,4}" \
  ARMS="${ARMS:-privileged-e4:privileged:4:candidate,control-e4:plain:4:control}" \
  SEED_RULE=mean-with-floor SEED_GATE="${SEED_GATE:-1}" EARLY_STOP=1 CRITIC_VALIDATION=1 \
  POPULATION_TAIL="${POPULATION_TAIL:-advisory}" \
  LR="${LR:-1e-4}" \
  FAMILY_WEIGHTS="${FAMILY_WEIGHTS:-default,under-clothed=2}" FAMILY_WEIGHT_DECAY=cosine \
  INIT_ARGS="${INIT_ARGS:---qol-encoder shared}" \
  CRITIC_ARGS="${CRITIC_ARGS:---qol-loss-weight 30 --qol-head-bias-init mean-target --td-step0-targets mc --tail-normalize world-factor --false-terminal-policy mask}" \
  QOL_REFERENCE_SCALE="$SCALE" \
  RESUME_EVERY="${RESUME_EVERY:-100}" DETERMINISM_STEPS="${DETERMINISM_STEPS:-50}" \
  GAP_FIT_DATA="${GAP_FIT_DATA:-}" CONTROL_REPORT="${CONTROL_REPORT:-}" \
    bash "$CODE/tools/r5_critic_acceptance.sh" >> "$WORK/critic.log" 2>&1 < /dev/null \
    || log "critic acceptance exited non-zero (see critic.log); uploading what exists"
  [ -f "$CRITIC_OUT/critic-receipt.json" ] || die "no critic-receipt.json (see critic.log, $CRITIC_OUT/driver.log)"
fi
cp "$WORK/critic.log" "$CRITIC_OUT/critic-queue.log" 2> /dev/null || true
log "critic receipt: $(grep -m1 -o '"machine_verdict": "[A-Z]*"' "$CRITIC_OUT/critic-receipt.json") $("$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); c=r.get("critic_stage") or {}; print("critic_stage", c.get("status"), {k: v["status"] for k, v in (c.get("arms") or {}).items()})' "$CRITIC_OUT/critic-receipt.json")"
touch "$WORK/DONE"
