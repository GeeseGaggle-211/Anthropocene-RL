#!/usr/bin/env bash
# r5 actor BC pilot (NOT gate evidence): independent observer BC (brief G2A) on the r5
# teacher batches fit8-b01..b04, one seed, the brief's schedule (batch 256, peak LR 3e-5,
# 10% warm-up, cosine to 10%, gradient cap 1), family weights default,under-clothed=2 with
# cosine decay, stability monitor with live early stop. Then same-state actor fidelity on
# the calibration batches and on the unseen fitting batch fit8-b05, and upload.
#
# Runs where it is started: DEVICE=cuda on the GPU VM (gpu_queue.sh calls it before the
# critic stage), DEVICE=cpu on the CPU VM. The development-panel rollouts (24 worlds x 50
# decisions) need the game and run on the CPU VM (cpu_queue.sh), from the uploaded actor.
#
# Resumable: a finished step is skipped by its output; an interrupted BC continues from
# <out>.partial (tools/g0launch.py, every G0_SAVE_EVERY steps) with the LR schedule continued
# at the partial's global step. Disclosed approximation on that path only: Adam moments, the
# shuffle stream and the family-weight decay restart at the partial (pilot, not evidence).
#
# Required env: CODE (learner checkout, .../rl), PY, DATA (dir holding fit8-b01..b05 and
# cal8-b01..b03), NORMALIZER (r5 actor normalizer), PRIV_NORMALIZER, OUTD.
# Optional: DEVICE=cuda EPOCHS=25 SEED=0 LR=3e-5 BATCH=256 BC_BATCHES="fit8-b01 fit8-b02
#   fit8-b03r fit8-b04" FIDELITY_BATCHES="cal8-b01 cal8-b02 cal8-b03 fit8-b05" THREADS=12
#   FAMILY_WEIGHTS=default,under-clothed=2 G0_SAVE_EVERY=500 UPLOAD=1
#   ANTHROPOCENE_DOTNET / ANTHROPOCENE_MODEL9_BRIDGE (initialize probes the bridge).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "$here/lib.sh"
STAGE_NAME=actor-bc
: "${CODE:?}" "${PY:?}" "${DATA:?}" "${NORMALIZER:?}" "${PRIV_NORMALIZER:?}" "${OUTD:?}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-25}"
SEED="${SEED:-0}"
LR="${LR:-3e-5}"
BATCH="${BATCH:-256}"
THREADS="${THREADS:-12}"
BC_BATCHES="${BC_BATCHES:-fit8-b01 fit8-b02 fit8-b03r fit8-b04}"
FIDELITY_BATCHES="${FIDELITY_BATCHES:-cal8-b01 cal8-b02 cal8-b03 fit8-b05}"
FAMILY_WEIGHTS="${FAMILY_WEIGHTS:-default,under-clothed=2}"
export G0_SAVE_EVERY="${G0_SAVE_EVERY:-500}"
UPLOAD="${UPLOAD:-1}"
DEST="${ACTOR_PILOT_GCS:-$SCALEOUT_GCS/actor-pilot}"

mkdir -p "$OUTD"
LOG_FILE="$OUTD/driver.log"
export TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 PYTHONPATH="$CODE/src"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
M=anthropocene_rl.autonomous_v7
raise_nofile > /dev/null
cd "$CODE"
[ -f "$OUTD/DONE" ] && { log "already done ($OUTD/DONE)"; exit 0; }
log "start device=$DEVICE epochs=$EPOCHS seed=$SEED lr=$LR batches=[$BC_BATCHES] code=$(cat "$CODE/../COMMIT" 2>/dev/null || git -C "$CODE" rev-parse --short HEAD 2>/dev/null || echo unknown)"

bc_dirs=(); for b in $BC_BATCHES; do require_file "$DATA/$b/dataset.json"; bc_dirs+=("$DATA/$b"); done

# 1. Fresh r5 checkpoint (the same initialize as the critic arms; the actor is trained here,
#    the critics stay untrained and are never used by this pilot).
init="$OUTD/checkpoint-000.pt"
if [ ! -f "$init" ]; then
  log "initialize"
  "$PY" -m $M.labor_training initialize --campaign-r5 --normalizer-checkpoint "$NORMALIZER" \
    --world-seed 90100 --optimizer-seed "$SEED" --qol-head logistic,td-lambda,0.8 \
    --qol-encoder shared --privileged-normalizer "$PRIV_NORMALIZER" --out "$init" \
    >> "$OUTD/initialize.log" 2>&1 < /dev/null || die "initialize (see initialize.log)"
fi

# 2. BC with the brief's schedule over the whole program (EPOCHS x steps/epoch).
rows=$("$PY" -c 'import json,sys; print(sum(s["rows"] for p in sys.argv[1:] for s in json.load(open(p+"/dataset.json"))["shards"].values()))' "${bc_dirs[@]}")
spe=$(( (rows + BATCH - 1) / BATCH ))
total=$(( EPOCHS * spe ))
actor="$OUTD/actor-bc-pilot.pt"
if [ ! -f "$actor" ]; then
  ck=$init; epochs=$EPOCHS; start=0
  if [ -f "$actor.partial" ]; then
    start=$("$PY" -c 'import torch,sys; d=torch.load(sys.argv[1],map_location="cpu",weights_only=False); print(int((d.get("g0_partial") or {}).get("steps",0)))' "$actor.partial")
    cp "$actor.partial" "$OUTD/actor.resume.pt"; ck=$OUTD/actor.resume.pt
    epochs=$(( EPOCHS - start / spe )); [ $epochs -lt 1 ] && epochs=1
    log "resume BC from partial at step $start of $total ($epochs epochs left; Adam restarts)"
  fi
  log "pretrain-bc rows=$rows steps/epoch=$spe total_steps=$total from $ck"
  args=(); for d in "${bc_dirs[@]}"; do args+=(--dataset "$d"); done
  "$PY" "$CODE/tools/g0launch.py" pretrain-bc --checkpoint "$ck" "${args[@]}" \
    --epochs "$epochs" --batch-size "$BATCH" --lr "$LR" --lr-schedule warmup-cosine \
    --lr-warmup-fraction 0.1 --lr-final-fraction 0.1 \
    --lr-schedule-total-steps "$total" --lr-schedule-start-step "$start" --max-grad-norm 1.0 \
    --family-weights "$FAMILY_WEIGHTS" --family-weight-decay cosine \
    --stability-log "$OUTD/stability-bc.jsonl" --stability-early-stop \
    --device "$DEVICE" --allow-vm-scale --out "$actor" \
    >> "$OUTD/bc.log" 2>&1 < /dev/null || die "pretrain-bc (see bc.log)"
  rm -f "$OUTD/actor.resume.pt"
  log "BC done: $(grep -o 's_per_step=[0-9.]*' "$OUTD/bc.log" | tail -1)"
fi

# 3. Same-state fidelity (CPU forward passes): calibration worlds and unseen fitting worlds.
for b in $FIDELITY_BATCHES; do
  out="$OUTD/fidelity-$b.json"
  [ -f "$out" ] && continue
  [ -f "$DATA/$b/dataset.json" ] || { log "skip fidelity on $b (not staged)"; continue; }
  log "actor-fidelity $b"
  "$PY" -m $M.labor_pretrain_eval actor-fidelity --checkpoint "$actor" --dataset "$DATA/$b" \
    --batch-size 256 --out "$out" > "$OUTD/fidelity-$b.log" 2>&1 < /dev/null \
    || log "actor-fidelity $b failed (see fidelity-$b.log); continuing"
done

sha256sum "$actor" > "$actor.sha256"
if [ "$UPLOAD" = "1" ]; then
  log "upload to $DEST"
  gcs_sync "$OUTD" "$DEST" '.*\.(partial|partial\.tmp)$|.*actor\.resume\.pt$'
fi
touch "$OUTD/DONE"
[ "$UPLOAD" = "1" ] && gcs_put "$OUTD/DONE" "$DEST/DONE"
log "done: $actor ($(cut -c1-16 "$actor.sha256"))"
