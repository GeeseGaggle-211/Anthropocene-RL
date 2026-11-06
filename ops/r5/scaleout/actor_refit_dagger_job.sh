#!/usr/bin/env bash
# r5 actor BC REFIT D1: R1's refit plus the actor-from-reset DAgger dataset
# (GPU VM anthropocene-gpu-fit-l4, g2-standard-16, 1 L4, STANDARD provisioning).
#
# Derived from the R1 refit script (actor_refit_job.sh, run outside this repo; its outputs and
# exact command are in gs://.../model9/r5/scaleout/actor-refit-r1/{bc-command.txt,config.json}).
#
# INTENDED DIFFERENCE FROM R1: the DAgger dataset is one more --dataset, and actor-from-reset
# DAgger rows are family-tagged (labor_scenarios.row_family_tags, code of $SHA), and nothing else.
#   R1 (code df1aac5, actor-bc-refit-r1.pt) ran pretrain-bc on fit8-b01 fit8-b02 fit8-b03r fit8-b04
#   (136,472 rows, 534 steps/epoch, 25 epochs = 13,350 steps), seed 0, batch 256, peak LR 3e-5
#   (warm-up 10%, cosine to 10%), gradient cap 1.0, --family-weights default,under-clothed=2 with
#   cosine decay, stability log + early stop, device cuda, --allow-vm-scale, BALANCE_ARGS
#   "--loss-balancing grouped --gradnorm-inner none --gradnorm-init sqrt
#   --gradnorm-inner-max-weight 100 --gradnorm-every 4", initialised from checkpoint-000.pt
#   (sha256 d48b3c06...), fidelity on cal8-b01 cal8-b02 cal8-b03 fit8-b05. This job runs the SAME
#   command with "--dataset <DAgger data>" appended after the four teacher batches. pretrain-bc
#   (labor_teacher_data._load_labor_datasets, no merge_teacher) then gives the four teacher
#   batches (all row_source 0) one source and the DAgger rows (row_source 1) the other: equal
#   source mass, equal mass per world within each source; family up-weights are rescaled inside
#   each (source, world) group, so they never move mass between sources or worlds. The LR
#   schedule spans EPOCHS x ceil(N / BATCH) steps with N = every teacher + DAgger row (pretrain-bc
#   iterates all rows, quarantined ones at weight 0), so it ends exactly at the last step. The
#   code is the DAgger branch tarball rl-dagger-$SHA (differs from R1's df1aac5 by design: it
#   adds the actor-from-reset tagging); config.json records everything above.
#
# Stages (each logged with UTC time and duration to driver.log / timings.tsv; a finished stage
# is skipped on rerun by its output, locally or in GCS):
#   preflight  GPU, torch CUDA, venv, disk, memory estimate, duplicate guards, DEST guards
#   code       gs://.../r5/code/rl-dagger-$SHA.tar.gz, commit id verified against $SHA, the code
#              must have pretrain-bc --loss-balancing, the BALANCE_ARGS flags and the
#              actor-from-reset tagging (labor_scenarios.dagger_row_prefixes)
#   data       the 4 teacher batches + fidelity sets (EXPECTED_ROWS teacher rows), normalizers
#              (sha256), and the DAgger dataset: refused unless $DAGGER_STATUS begins with "0";
#              fetched from $DAGGER_GCS; dataset.json present, every shard present and non-empty,
#              source record = learner_state_dagger / prefix actor-from-reset / collected with
#              the R1 actor (DAGGER_ACTOR_SHA256) on teacher-batch worlds
#   init       R1's checkpoint-000.pt from $INIT_GCS (sha256 must equal INIT_SHA256)
#   datacheck  loads the BC datasets exactly as pretrain-bc does (same loader call, family
#              weights, checkpoint normalizer) and writes datacheck.json: N (must equal the
#              shard-row sum), sources and mass, the DAgger component prefix, the family-weighting
#              record (shares before/after, overall and per source), peak RSS of the load
#   bc         tools/g0launch.py pretrain-bc ... (resumes from <out>.partial); afterwards the
#              checkpoint's lr_schedule total_steps must equal EPOCHS x ceil(N / BATCH)
#   loadcheck  torch.load(weights_only=True) of the result (what the panel evaluator does)
#   fidelity   actor-fidelity on cal8-b01 cal8-b02 cal8-b03 fit8-b05, as R1
#   upload     every result object once (no-clobber), config.json once, then EXIT_STATUS last
#
# GCS objects are WRITE-ONCE (the VM service account cannot overwrite or delete objects):
#   * $DEST/EXIT_STATUS is uploaded exactly once, last ("0 <utc> ..." or "FAIL:<stage> <utc>").
#     If it already exists at start the job logs and exits without doing anything (no shutdown):
#     a FAIL finalises this DEST, relaunch with a new DEST. Likewise, while $DAGGER_STATUS does
#     not begin with "0" the job does not start (exit 5, nothing written, no shutdown).
#   * An interrupted launch (TERM/INT/HUP, e.g. a VM stop) uploads no EXIT_STATUS: rerun the same
#     line; BC resumes from <out>.partial.
#   * A refused configuration (DEST/WORK of another run id, memory estimate) uploads no
#     EXIT_STATUS either, so the DEST stays usable.
#   * $DEST/RUNNING (start marker) and $DEST/RUN_ID are uploaded only when absent; a different
#     remote RUN_ID refuses this DEST. Per-launch objects use fresh names: config-<launch>.json,
#     logs/driver-<launch>.log, logs/launch-<launch>.out, progress/<launch>/<utc>-*.
#   * Results (actor, .sha256, bc.log, bc-command.txt, stability-bc.jsonl, datacheck.json,
#     timings.tsv, fidelity-*, resume_events.jsonl) and config.json are uploaded only when absent.
# The EXIT trap uploads the launch log, writes EXIT_STATUS and powers the VM off on success, on
# failure and on interruption, unless NO_SHUTDOWN=1 (or the file $WORK/NO_SHUTDOWN exists:
# `touch` it to keep the VM up after the job was already launched). A refused duplicate launch
# (lock/worker guard, finished DEST) never shuts anything down.
#
# Memory: pretrain-bc concatenates the loaded parts into one dataset, so the load peaks at about
# twice the BC data size in anonymous memory (R1: ~2 x 13 GiB; with the DAgger rows about twice
# that). Preflight refuses (no EXIT_STATUS) when 2 x the npy bytes + MEM_HEADROOM_GB exceeds
# MemTotal (MEM_CHECK=0 skips it); datacheck records the measured peak RSS.
#
# Resume caveat (as R1's, disclosed): an interrupted BC continues from <out>.partial (g0launch,
# every G0_SAVE_EVERY steps) with the LR schedule continued at the partial's global step; Adam
# moments, the shuffle stream, the family-weight decay AND the GradNorm controller state restart
# at the partial. A resumed run is recorded in config.json (resume_events, uninterrupted=false).
#
# Required env:
#   SHA            code commit (full 40-hex) of gs://.../r5/code/rl-dagger-$SHA.tar.gz
# Optional env (defaults = R1): DEVICE=cuda EPOCHS=25 SEED=0 LR=3e-5 BATCH=256 BC_THREADS=12
#   BC_BATCHES="fit8-b01 fit8-b02 fit8-b03r fit8-b04" FIDELITY_BATCHES="cal8-b01 cal8-b02 cal8-b03
#   fit8-b05" FAMILY_WEIGHTS=default,under-clothed=2 EXPECTED_ROWS=136472 (teacher rows)
#   BALANCE_ARGS="<R1's, above>" G0_SAVE_EVERY=500 BC_TIME_FACTOR=1.0 (ETA log only: R1 0.305
#   s/step x factor) BC_TIMEOUT=<s> NO_SHUTDOWN=1 FRESH=1 (archive a WORK dir of a different
#   configuration) MEM_CHECK=0 MEM_HEADROOM_GB=3 MIN_DISK_GB=40
#   DAGGER_GCS=gs://.../scaleout/actor-dagger-r1/data DAGGER_STATUS=<DAGGER_GCS minus /data>/EXIT_STATUS
#   DAGGER_ACTOR_SHA256=<R1 actor sha256> EXPECTED_DAGGER_ROWS=<pin the DAgger row count; unset: any>
#   INIT_GCS=gs://.../scaleout/actor-refit-r1 INIT_SHA256=d48b3c06...
#   DEST=gs://.../scaleout/actor-refit-d1 ACTOR_NAME=actor-bc-refit-d1.pt WORK=$HOME/r5gpu/actor-refit-d1
#   R5GPU=$HOME/r5gpu DATA=$R5GPU/pilot DAGGER_DIR=$R5GPU/dagger/<DAgger run name> PY=$R5GPU/venv/bin/python
#
# Launch on the GPU VM (detached; survives the SSH session; rerun the identical line after a stop):
#   SHA=<sha> setsid nohup bash $HOME/actor_refit_dagger_job.sh \
#     > $HOME/r5gpu/actor-refit-d1.out 2>&1 < /dev/null &
# Code tarball (operator machine, in the r5-takeover-rl worktree; not run by this job):
#   SHA=$(git -C /c/Users/suyas/Documents/ChatGPT/worktrees/r5-takeover-rl rev-parse HEAD) \
#   && git -C /c/Users/suyas/Documents/ChatGPT/worktrees/r5-takeover-rl -c core.autocrlf=false \
#        archive --format=tar.gz -o "rl-dagger-$SHA.tar.gz" "$SHA" \
#   && gcloud storage cp "rl-dagger-$SHA.tar.gz" \
#        "gs://anthropocene-rl-datasets-508912/model9/r5/code/rl-dagger-$SHA.tar.gz"
set -euo pipefail

SHA="${SHA:?set SHA to the code commit of gs://.../r5/code/rl-dagger-<sha>.tar.gz}"
R1_BALANCE_ARGS="--loss-balancing grouped --gradnorm-inner none --gradnorm-init sqrt --gradnorm-inner-max-weight 100 --gradnorm-every 4"
BALANCE_ARGS="${BALANCE_ARGS:-$R1_BALANCE_ARGS}"
[ -n "${BALANCE_ARGS//[[:space:]]/}" ] || { echo "BALANCE_ARGS is blank" >&2; exit 2; }
read -r -a balance_args <<< "$BALANCE_ARGS"

GCS_ROOT="${GCS_ROOT:-gs://anthropocene-rl-datasets-508912/model9/r5}"
CODE_GCS="${CODE_GCS:-$GCS_ROOT/code/rl-dagger-$SHA.tar.gz}"
R1_GCS="$GCS_ROOT/scaleout/actor-refit-r1"
INIT_GCS="${INIT_GCS:-$R1_GCS}"
INIT_SHA256="${INIT_SHA256:-d48b3c06381b500398244a133eb39447d38811b2bb0736bbaeb27ce920aeb7e8}"
DAGGER_GCS="${DAGGER_GCS:-$GCS_ROOT/scaleout/actor-dagger-r1/data}"
DAGGER_GCS="${DAGGER_GCS%/}"
DAGGER_STATUS="${DAGGER_STATUS:-${DAGGER_GCS%/data}/EXIT_STATUS}"
DAGGER_ACTOR_SHA256="${DAGGER_ACTOR_SHA256:-f1f4076fc4a69e901cc21811d30c3708a89470d58c3b47571a7af556be02957d}"
EXPECTED_DAGGER_ROWS="${EXPECTED_DAGGER_ROWS:-}"
DEST="${DEST:-$GCS_ROOT/scaleout/actor-refit-d1}"
DEST="${DEST%/}"
R="${R5GPU:-$HOME/r5gpu}"
WORK="${WORK:-$R/actor-refit-d1}"
OUTD="$WORK"
DATA="${DATA:-$R/pilot}"
DAGGER_RUN_NAME=$(basename "${DAGGER_GCS%/data}")
DAGGER_DIR="${DAGGER_DIR:-$R/dagger/$DAGGER_RUN_NAME}"
CODEROOT="${CODEROOT:-$R/code-dagger-$SHA}"
CODE="$CODEROOT/rl"
PY="${PY:-$R/venv/bin/python}"
ACTOR_NAME="${ACTOR_NAME:-actor-bc-refit-d1.pt}"
ACTOR_NORMALIZER_SHA256="3c8c5ca300bab11d9651dfc4bfbaa36c7bae08e4c0236773c84e2237f1fa57de"
R1_ACTOR_SHA256="f1f4076fc4a69e901cc21811d30c3708a89470d58c3b47571a7af556be02957d"
R1_CODE="df1aac532b3d5e270a079e300462eabe7cff1d28"
R1_S_PER_STEP="0.305"

# R1 configuration.
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-25}"
SEED="${SEED:-0}"
LR="${LR:-3e-5}"
BATCH="${BATCH:-256}"
THREADS="${BC_THREADS:-12}"
BC_BATCHES="${BC_BATCHES:-fit8-b01 fit8-b02 fit8-b03r fit8-b04}"
FIDELITY_BATCHES="${FIDELITY_BATCHES:-cal8-b01 cal8-b02 cal8-b03 fit8-b05}"
FAMILY_WEIGHTS="${FAMILY_WEIGHTS:-default,under-clothed=2}"
EXPECTED_ROWS="${EXPECTED_ROWS:-136472}"
BC_TIME_FACTOR="${BC_TIME_FACTOR:-1.0}"
export G0_SAVE_EVERY="${G0_SAVE_EVERY:-500}"
M=anthropocene_rl.autonomous_v7

ts() { date -u +%FT%TZ; }
LAUNCH=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$WORK/logs"
LOG_FILE="$WORK/driver.log"
LAUNCH_LOG="$WORK/logs/driver-$LAUNCH.log"
CUR_STAGE=startup
log() { echo "$(ts) [actor-refit-d1] $*" | tee -a "$LOG_FILE" "$LAUNCH_LOG" >&2; }
die() { log "FAIL[$CUR_STAGE] $*"; exit 1; }
sha256_of() { sha256sum "$1" | cut -d' ' -f1; }
remote_has() { timeout 120 gcloud storage ls "$1" > /dev/null 2>&1 < /dev/null; }
gcs_put() { timeout 600 gcloud storage cp --no-user-output-enabled "$1" "$2" < /dev/null || log "upload failed: $1 -> $2"; }
# Write-once: an object already present is never re-uploaded (the service account cannot overwrite).
gcs_put_once() { remote_has "$2" && { log "present, not re-uploaded: $2"; return 0; }; timeout 600 gcloud storage cp --no-user-output-enabled "$1" "$2" < /dev/null; }
require_file() { [ -e "$1" ] || die "missing $1${2:+ ($2)}"; }
disk_free_gb() { df -Pk "$1" | awk 'NR==2{print int($4/1048576)}'; }

# Duplicate guards (before the EXIT trap: a refused duplicate must not upload, write
# EXIT_STATUS or stop the VM). The [x] brackets keep pgrep from matching its own command line
# (and this script's, whose argv never contains the worker command names).
exec 9> "$WORK/job.lock"
flock -n 9 || { log "another actor_refit_dagger_job.sh holds $WORK/job.lock; exiting without touching anything"; exit 3; }
if pgrep -f '[g]0launch\.py pretrain-' > /dev/null; then
  log "a g0launch pretrain process is already running: $(pgrep -af '[g]0launch\.py pretrain-' | head -3); exiting without touching anything"
  exit 3
fi
# A final EXIT_STATUS cannot be replaced (no delete permission): this DEST is finished.
if remote_has "$DEST/EXIT_STATUS"; then
  log "$DEST/EXIT_STATUS already exists ($(timeout 120 gcloud storage cat "$DEST/EXIT_STATUS" < /dev/null 2> /dev/null | head -1)); nothing to do (use a new DEST to rerun)"
  exit 0
fi
# The DAgger data must come from a finished collection: refuse to start (nothing written, the VM
# left running) unless its EXIT_STATUS begins with "0". Rechecked and recorded in stage_data.
dagger_status=$(timeout 120 gcloud storage cat "$DAGGER_STATUS" < /dev/null 2> /dev/null | head -1 || true)
if [ "${dagger_status%% *}" != "0" ]; then
  log "DAgger collection status is '${dagger_status:-absent}' at $DAGGER_STATUS (needs to begin with 0): not starting, nothing written (rerun when the collection has finished)"
  exit 5
fi

# Configuration identity: a WORK dir / DEST of a different configuration is never reused.
RUN_ID=$(printf '%s\n' "$SHA" "$BALANCE_ARGS" "$EPOCHS" "$SEED" "$LR" "$BATCH" "$FAMILY_WEIGHTS" \
  "$BC_BATCHES" "$DEVICE" "$INIT_GCS" "$INIT_SHA256" "$DAGGER_GCS" "$DAGGER_ACTOR_SHA256" | sha256sum | cut -c1-16)
if [ -f "$WORK/RUN_ID" ] && [ "$(cat "$WORK/RUN_ID")" != "$RUN_ID" ]; then
  if [ "${FRESH:-0}" = "1" ]; then
    old="$WORK.old-$(date -u +%Y%m%dT%H%M%SZ)"
    log "FRESH=1: archiving the different-configuration WORK dir to $old"
    flock -u 9; exec 9>&-
    mv "$WORK" "$old"; mkdir -p "$WORK/logs"; exec 9> "$WORK/job.lock"; flock -n 9 || exit 3
    cp "$old/logs/driver-$LAUNCH.log" "$LAUNCH_LOG" 2> /dev/null || true
  else
    log "$WORK holds run $(cat "$WORK/RUN_ID") but this launch is $RUN_ID (different SHA/settings); set FRESH=1 to archive it, or use another WORK/DEST"
    exit 4
  fi
fi
echo "$RUN_ID" > "$WORK/RUN_ID"

shutdown_vm() {
  if [ "${NO_SHUTDOWN:-0}" = "1" ] || [ -e "$WORK/NO_SHUTDOWN" ]; then
    log "NO_SHUTDOWN: leaving the VM running"
    return 0
  fi
  log "shutting the VM down"
  sync
  sudo -n shutdown -h now "r5 actor refit d1 finished" || log "shutdown failed (sudo)"
}

# Result objects uploaded (once) by stage_upload; a failed launch uploads what exists.
RESULT_FILES=(bc.log bc-command.txt stability-bc.jsonl datacheck.json datacheck.log bc-schedule.json timings.tsv resume_events.jsonl RUN_ID)
upload_results() {  # every present result once; returns 1 when an upload failed
  local f rc=0 extra=()
  for f in "$WORK"/fidelity-*.json "$WORK"/fidelity-*.log; do [ -f "$f" ] && extra+=("${f##*/}"); done
  for f in "${RESULT_FILES[@]}" ${extra[@]+"${extra[@]}"}; do
    [ -f "$WORK/$f" ] || continue
    gcs_put_once "$WORK/$f" "$DEST/$f" || { log "upload failed: $f"; rc=1; }
  done
  return "$rc"
}

SKIP_STATUS=0   # 1: this launch must not upload EXIT_STATUS (interrupted, or configuration refused)
DEST_FOREIGN=0  # 1: DEST belongs to another run: nothing at all is uploaded to it
uploader_pid=""
finish() {
  local status=$?
  trap - EXIT
  set +e
  [ -n "$uploader_pid" ] && kill "$uploader_pid" 2> /dev/null
  stage_end "$status"
  if [ "$SKIP_STATUS" != "1" ]; then
    if [ "$status" -eq 0 ]; then
      log "refit job finished"
      [ -f "$WORK/EXIT_STATUS" ] || echo "0 $(ts)" > "$WORK/EXIT_STATUS"
    else
      log "refit job FAILED status=$status stage=$CUR_STAGE"
      echo "FAIL:$CUR_STAGE $(ts)" > "$WORK/EXIT_STATUS"
      upload_results   # whatever exists, once (a FAIL finalises this DEST)
    fi
  else
    log "no EXIT_STATUS from this launch (interrupted or refused); status=$status"
  fi
  if [ "$DEST_FOREIGN" = "1" ]; then
    log "configuration conflict: nothing uploaded, DEST untouched"
    shutdown_vm
    exit "$status"
  fi
  [ -f "$R/actor-refit-d1.out" ] && gcs_put "$R/actor-refit-d1.out" "$DEST/logs/launch-$LAUNCH.out"
  gcs_put "$LAUNCH_LOG" "$DEST/logs/driver-$LAUNCH.log"
  if [ "$SKIP_STATUS" != "1" ] && [ -f "$WORK/EXIT_STATUS" ]; then
    if remote_has "$DEST/EXIT_STATUS"; then
      log "$DEST/EXIT_STATUS appeared during this launch; not overwriting (local: $(cat "$WORK/EXIT_STATUS"))"
    else
      gcs_put "$WORK/EXIT_STATUS" "$DEST/EXIT_STATUS"   # last and once
    fi
  fi
  shutdown_vm
  exit "$status"
}
trap finish EXIT
trap 'SKIP_STATUS=1; log "signal received: interrupted, no EXIT_STATUS (rerun the same line to resume)"; exit 143' TERM INT HUP

STAGE_T0=0
STAGE_OPEN=0
stage_end() {  # stage_end RC : close the open stage (timings.tsv + log)
  local d=$(( $(date +%s) - STAGE_T0 ))
  [ "$STAGE_OPEN" = "1" ] || return 0
  STAGE_OPEN=0
  printf '%s\t%s\t%s\t%s\n' "$CUR_STAGE" "$(ts)" "$d" "$1" >> "$WORK/timings.tsv"
  log "END $CUR_STAGE rc=$1 in ${d} s"
}
timed() {  # timed NAME command... : a fatal stage run in this shell (set -e ends the job; the trap records it)
  CUR_STAGE=$1; STAGE_T0=$(date +%s); STAGE_OPEN=1
  shift
  log "BEGIN $CUR_STAGE"
  "$@"
  stage_end 0
}
timed_soft() {  # timed_soft NAME command... : like timed, but returns the command's status
  local rc=0
  CUR_STAGE=$1; STAGE_T0=$(date +%s); STAGE_OPEN=1
  shift
  log "BEGIN $CUR_STAGE"
  "$@" || rc=$?
  stage_end "$rc"
  return "$rc"
}

# --- stages -------------------------------------------------------------------------------
stage_preflight() {
  [ "$DEVICE" != "cuda" ] || nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader >> "$LOG_FILE" 2>&1 \
    || die "nvidia-smi failed"
  require_file "$PY" "venv"
  if [ "$DEVICE" = "cuda" ]; then
    "$PY" - <<'PY' >> "$LOG_FILE" 2>&1 || die "torch CUDA unavailable"
import torch
assert torch.cuda.is_available()
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
PY
  fi
  local need="${MIN_DISK_GB:-40}" free
  free=$(disk_free_gb "$R")
  [ "$free" -ge "$need" ] || log "WARN only ${free} GB free under $R (want ${need} to stage missing datasets; the DAgger data is ~15 GB)"
  [ "$free" -ge 10 ] || die "only ${free} GB free under $R"
  log "disk free ${free} GB; host $(hostname); run_id=$RUN_ID launch=$LAUNCH; memory: $(free -g | awk 'NR==2{print $2" GiB total, "$7" GiB available"} /^Swap/{print "swap "$2" GiB"}' | tr '\n' ' ')"
  local hard
  hard=$(ulimit -Hn); ulimit -Sn "$hard" 2> /dev/null || ulimit -Sn 65536 2> /dev/null || true
  log "open-file limit soft=$(ulimit -Sn) hard=$(ulimit -Hn)"
  # The DEST of another configuration is never written to (RUN_ID is uploaded once, below).
  local remote_id
  remote_id=$(timeout 120 gcloud storage cat "$DEST/RUN_ID" < /dev/null 2> /dev/null | head -1 || true)
  if [ -n "$remote_id" ] && [ "$remote_id" != "$RUN_ID" ]; then
    SKIP_STATUS=1; DEST_FOREIGN=1
    die "$DEST holds run $remote_id, this launch is $RUN_ID; use another DEST"
  fi
  [ -n "$remote_id" ] || gcs_put_once "$WORK/RUN_ID" "$DEST/RUN_ID" || die "cannot upload $DEST/RUN_ID"
  echo "RUNNING $(ts) launch=$LAUNCH run_id=$RUN_ID sha=$SHA" > "$WORK/RUNNING"
  remote_has "$DEST/RUNNING" || gcs_put "$WORK/RUNNING" "$DEST/RUNNING"   # start marker, once
}

tar_commit_id() {  # tar_commit_id TARBALL -> the git archive commit id (empty when none)
  local id=""
  if command -v git > /dev/null 2>&1; then
    # get-tar-commit-id stops reading after the header, so gzip may die of SIGPIPE (141)
    # under pipefail; judge by the output, not the pipeline status.
    id=$(gzip -dc "$1" | git get-tar-commit-id 2> /dev/null || true)
  fi
  if [ -z "$id" ]; then
    id=$("$PY" -c 'import sys,tarfile; t=tarfile.open(sys.argv[1]); t.next(); print(t.pax_headers.get("comment",""))' "$1" 2> /dev/null || true)
  fi
  echo "$id"
}

stage_code() {
  if [ ! -f "$CODEROOT/.snapshot-ok" ]; then
    log "fetching $CODE_GCS"
    rm -rf "$CODEROOT" "$CODEROOT.tmp"
    mkdir -p "$CODEROOT.tmp/rl"
    timeout 600 gcloud storage cp --no-user-output-enabled "$CODE_GCS" "$CODEROOT.tmp/code.tar.gz" < /dev/null \
      || die "download failed: $CODE_GCS (build and upload the tarball first)"
    local archived
    archived=$(tar_commit_id "$CODEROOT.tmp/code.tar.gz")
    [ -n "$archived" ] || die "tarball has no commit id (was it built with git archive <commit>?)"
    [ "$archived" = "$SHA" ] || [ "${archived:0:${#SHA}}" = "$SHA" ] || die "tarball commit $archived != SHA $SHA"
    tar -xzf "$CODEROOT.tmp/code.tar.gz" -C "$CODEROOT.tmp/rl"
    echo "$archived" > "$CODEROOT.tmp/COMMIT"
    sha256_of "$CODEROOT.tmp/code.tar.gz" > "$CODEROOT.tmp/TARBALL_SHA256"
    if command -v git > /dev/null 2>&1; then  # best-effort local snapshot commit (provenance only)
      (cd "$CODEROOT.tmp/rl" && git init -q && git add -A -f \
        && git -c user.name=refit -c user.email=refit@localhost commit -q -m "snapshot of $archived (git archive)") \
        > /dev/null 2>&1 || log "WARN snapshot commit failed (provenance falls back to unknown)"
    fi
    mv "$CODEROOT.tmp" "$CODEROOT"
    touch "$CODEROOT/.snapshot-ok"
  fi
  export TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 PYTHONPATH="$CODE/src"
  export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
  require_file "$CODE/tools/g0launch.py" "code tarball"
  local where
  where=$("$PY" -c 'import anthropocene_rl,sys; print(anthropocene_rl.__file__)' 2>&1) || die "cannot import anthropocene_rl: $where"
  case "$where" in "$CODE"/src/*) ;; *) die "anthropocene_rl resolves to $where, not $CODE/src" ;; esac
  local help_text tok tagging
  help_text=$(cd "$CODE" && "$PY" -m $M.labor_training pretrain-bc --help < /dev/null 2>&1 || true)
  case "$help_text" in *--loss-balancing*) ;; *) die "code $SHA lacks pretrain-bc --loss-balancing" ;; esac
  for tok in "${balance_args[@]}"; do
    case "$tok" in --*) grep -q -- "$tok" <<< "$help_text" || die "BALANCE_ARGS flag $tok is not a pretrain-bc option in this code" ;; esac
  done
  tagging=$(cd "$CODE" && "$PY" -c 'from anthropocene_rl.autonomous_v7 import labor_scenarios as S
assert hasattr(S, "dagger_row_prefixes") and S.DAGGER_PREFIX_ACTOR_FROM_RESET == "actor-from-reset"
print("ok")' < /dev/null 2>&1) || die "code $SHA lacks the actor-from-reset DAgger family tagging: $tagging"
  log "code $(cat "$CODEROOT/COMMIT") tarball sha256 $(cut -c1-16 "$CODEROOT/TARBALL_SHA256"); balance args: ${balance_args[*]}"
}

fetch_dataset() {  # fetch_dataset NAME : incremental, .fetched marker like gpu_queue.sh's gcs_fetch_dir
  local b=$1 dest="$DATA/$1"
  if [ -f "$dest/dataset.json" ] && [ -f "$dest/.fetched" ]; then return 0; fi
  log "staging $b"
  mkdir -p "$dest"
  timeout 7200 gcloud storage rsync -r --no-user-output-enabled "$GCS_ROOT/pilot/$b" "$dest" < /dev/null || die "download failed: $b"
  [ -f "$dest/dataset.json" ] || die "no dataset.json in $b"
  touch "$dest/.fetched"
}

shard_rows() {  # shard_rows DIR -> its row count; every shard directory must exist and be non-empty
  "$PY" - "$1" <<'PY' || die "dataset $1 failed the integrity check"
import json, os, sys
root = sys.argv[1]
meta = json.load(open(os.path.join(root, "dataset.json")))
rows = 0
for seed, shard in sorted(meta["shards"].items()):
    path = os.path.join(root, shard["path"])
    if not os.path.isdir(path) or not os.listdir(path):
        sys.exit(f"shard {seed} missing or empty: {path}")
    rows += int(shard["rows"])
print(rows)
PY
}

stage_data() {
  mkdir -p "$DATA"
  local b rows total=0 seen=""
  for b in $BC_BATCHES $FIDELITY_BATCHES; do
    case " $seen " in *" $b "*) continue ;; esac
    seen="$seen $b"
    fetch_dataset "$b"
    rows=$(shard_rows "$DATA/$b")
    case " $BC_BATCHES " in *" $b "*) total=$(( total + rows )) ;; esac
  done
  [ "$total" -eq "$EXPECTED_ROWS" ] || die "teacher BC batches hold $total rows, expected $EXPECTED_ROWS (R1's 136,472)"
  TEACHER_ROWS=$total
  log "teacher datasets present and verified; teacher BC rows=$TEACHER_ROWS"
  NORMALIZER="${NORMALIZER:-$DATA/actor-normalizer.pt}"
  PRIV_NORMALIZER="${PRIV_NORMALIZER:-$DATA/privileged-normalizer-r5.pt}"
  [ -f "$NORMALIZER" ] || timeout 600 gcloud storage cp "$GCS_ROOT/pilot/actor-normalizer.pt" "$NORMALIZER" < /dev/null
  [ -f "$PRIV_NORMALIZER" ] || timeout 600 gcloud storage cp "$GCS_ROOT/post-collection/privileged-normalizer.pt" "$PRIV_NORMALIZER" < /dev/null
  [ "$(sha256_of "$NORMALIZER")" = "$ACTOR_NORMALIZER_SHA256" ] || die "actor normalizer sha256 mismatch"
  log "normalizers ok (privileged sha256 $(sha256_of "$PRIV_NORMALIZER" | cut -c1-16))"

  # The DAgger dataset: only from a finished (status 0) collection.
  local status_text first
  status_text=$(timeout 120 gcloud storage cat "$DAGGER_STATUS" < /dev/null 2> /dev/null | head -1 || true)
  first=${status_text%% *}
  [ "$first" = "0" ] || die "DAgger collection status is '${status_text:-absent}' at $DAGGER_STATUS (needs to begin with 0); refusing to train on it"
  log "DAgger collection status: $status_text"
  echo "$status_text" > "$WORK/DAGGER_STATUS"
  if [ ! -f "$DAGGER_DIR/dataset.json" ] || [ ! -f "$DAGGER_DIR/.fetched" ]; then
    log "staging $DAGGER_GCS -> $DAGGER_DIR"
    mkdir -p "$DAGGER_DIR"
    timeout 7200 gcloud storage rsync -r --no-user-output-enabled "$DAGGER_GCS" "$DAGGER_DIR" < /dev/null || die "download failed: $DAGGER_GCS"
    [ -f "$DAGGER_DIR/dataset.json" ] || die "no dataset.json in $DAGGER_GCS"
    touch "$DAGGER_DIR/.fetched"
  fi
  DAGGER_ROWS=$(shard_rows "$DAGGER_DIR")
  [ "$DAGGER_ROWS" -gt 0 ] || die "the DAgger dataset has no rows"
  [ -z "$EXPECTED_DAGGER_ROWS" ] || [ "$DAGGER_ROWS" -eq "$EXPECTED_DAGGER_ROWS" ] || die "DAgger dataset holds $DAGGER_ROWS rows, expected $EXPECTED_DAGGER_ROWS"
  # Source record: an actor-from-reset DAgger collection of the R1 actor on teacher-batch worlds.
  local bc_dirs=() d
  for b in $BC_BATCHES; do bc_dirs+=("$DATA/$b"); done
  "$PY" - "$DAGGER_DIR" "$DAGGER_ACTOR_SHA256" "${bc_dirs[@]}" <<'PY' >> "$LOG_FILE" 2>&1 || die "the DAgger dataset is not an actor-from-reset collection of the R1 actor on the teacher worlds (see driver.log)"
import json, os, sys
dagger, actor_sha, teachers = sys.argv[1], sys.argv[2], sys.argv[3:]
manifest = json.load(open(os.path.join(dagger, "dataset.json")))
record = manifest["source_record"]
options = record["options"]
assert record.get("source_kind") == "learner_state_dagger", record.get("source_kind")
assert options.get("prefix") == "actor-from-reset", options.get("prefix")
assert options.get("checkpoint_sha256") == actor_sha, (options.get("checkpoint_sha256"), actor_sha)
worlds = {int(w) for w in manifest["shards"]}
teacher_worlds = {int(w) for t in teachers for w in json.load(open(os.path.join(t, "dataset.json")))["shards"]}
assert worlds <= teacher_worlds, sorted(worlds - teacher_worlds)
assert not any(90154 <= w <= 90159 for w in worlds), sorted(worlds)
for entry in manifest["shards"].values():
    meta = json.load(open(os.path.join(dagger, entry["path"], "meta.json")))
    assert meta.get("prefix") == "actor-from-reset", (entry["path"], meta.get("prefix"))
print(json.dumps({"dagger_source_check": "ok", "worlds": len(worlds), "dataset_digest": manifest["dataset_digest"],
                  "source_run": manifest["source_run"], "actor_mode": options.get("actor_mode"),
                  "starts_per_world": options.get("starts_per_world"), "turns": options.get("turns")}))
PY
  N_ROWS=$(( TEACHER_ROWS + DAGGER_ROWS ))
  log "DAgger dataset $DAGGER_DIR verified: rows=$DAGGER_ROWS; BC rows N=$N_ROWS (teacher $TEACHER_ROWS + DAgger $DAGGER_ROWS)"
  # Memory estimate: the aggregated torch.cat holds the loaded parts plus their concatenation.
  local data_kib mem_kib need_kib
  data_kib=$(du -sk --apparent-size "${bc_dirs[@]}" "$DAGGER_DIR" | awk '{s+=$1} END{print s}')
  mem_kib=$(awk '/MemTotal/{print $2}' /proc/meminfo)
  need_kib=$(( 2 * data_kib + ${MEM_HEADROOM_GB:-3} * 1048576 ))
  log "BC data $(( data_kib / 1048576 )) GiB on disk; estimated load peak $(( need_kib / 1048576 )) GiB; MemTotal $(( mem_kib / 1048576 )) GiB"
  if [ "$need_kib" -gt "$mem_kib" ]; then
    if [ "${MEM_CHECK:-1}" = "0" ]; then
      log "WARN estimated load peak exceeds MemTotal; MEM_CHECK=0, continuing"
    else
      SKIP_STATUS=1
      die "estimated load peak $(( need_kib / 1048576 )) GiB > MemTotal $(( mem_kib / 1048576 )) GiB: refused before any training (no EXIT_STATUS; DEST stays usable). Use a larger VM, add swap, or MEM_CHECK=0"
    fi
  fi
}

stage_init() {
  init="$OUTD/checkpoint-000.pt"
  if [ -f "$init" ]; then log "init present locally"
  else
    timeout 600 gcloud storage cp --no-user-output-enabled "$INIT_GCS/checkpoint-000.pt" "$init.tmp" < /dev/null \
      && mv "$init.tmp" "$init" || die "cannot fetch $INIT_GCS/checkpoint-000.pt"
  fi
  INIT_SHA256_ACTUAL=$(sha256_of "$init")
  [ "$INIT_SHA256_ACTUAL" = "$INIT_SHA256" ] || die "init checkpoint sha256 $INIT_SHA256_ACTUAL != R1's $INIT_SHA256"
  log "init checkpoint sha256 $INIT_SHA256_ACTUAL (R1's checkpoint-000.pt from $INIT_GCS)"
}

stage_datacheck() {
  # pretrain-bc's own loader call (labor_training._command_pretrain_bc), on the same directories
  # in the same order, with the same family weights and the checkpoint's normalizer.
  local out="$OUTD/datacheck.json"
  if [ -f "$out" ]; then log "datacheck present locally"; else
    local dirs=() b
    for b in $BC_BATCHES; do dirs+=("$DATA/$b"); done
    dirs+=("$DAGGER_DIR")
    (cd "$CODE" && "$PY" - "$init" "$FAMILY_WEIGHTS" "$N_ROWS" "$BATCH" "$EPOCHS" "$out.tmp" "${dirs[@]}" <<'PY' >> "$OUTD/datacheck.log" 2>&1) \
      || die "the BC datasets do not load as pretrain-bc loads them (see datacheck.log)"
import json, math, resource, sys, time
from pathlib import Path
import torch
from anthropocene_rl.autonomous_v7 import labor_training as lt
from anthropocene_rl.autonomous_v7.labor_scenarios import parse_family_weights
from anthropocene_rl.autonomous_v7.labor_teacher_data import load_labor_datasets

ckpt, weights, n_expected, batch, epochs, out = sys.argv[1:7]
dirs = [Path(d) for d in sys.argv[7:]]
state = torch.load(ckpt, map_location="cpu", weights_only=True)
_, _, _, _, normalizer = lt._models_from_checkpoint(state)
started = time.time()
dataset = load_labor_datasets(
    dirs,
    expected_normalizer_digest=lt._normalization_digest(normalizer),
    drop_reserved=False,
    family_weights=parse_family_weights(weights),
    family_normalizer=normalizer,
)
seconds = time.time() - started
n = int(dataset["target_specialty"].shape[0])  # pretrain_labor_actor_bc's N
assert n == int(n_expected), f"loaded N={n} != shard rows {n_expected}"
collection = dataset["collection"]
components = collection["component_collections"]
dagger = [c for c in components if c.get("source_kind") == "learner_state_dagger"]
assert len(dagger) == 1 and dagger[0].get("prefix") == "actor-from-reset", [c.get("prefix") for c in dagger]
source = dataset["row_source"].reshape(-1)
keep = ~dataset["row_quarantined"].reshape(-1)
tags = dataset["row_family_tags"].reshape(-1)
weight = dataset["row_weight"].double()
base = dataset["row_weight_unweighted"].double()
mass = {str(s): float(weight[source == s].sum()) for s in (0, 1)}
base_mass = {str(s): float(base[source == s].sum()) for s in (0, 1)}
assert math.isclose(base_mass["0"], base_mass["1"], rel_tol=1e-4), base_mass  # equal source mass
assert all(math.isclose(mass[k], base_mass[k], rel_tol=1e-4) for k in mass), (mass, base_mass)
tagged_dagger = int(((source == 1) & keep & (tags != 0)).sum())
assert tagged_dagger > 0, "no DAgger row is family-tagged"
steps = -(-n // int(batch))
record = collection["family_weighting"]
json.dump({
    "schema": "r5-actor-refit-dagger-datacheck-v1",
    "rows": n,
    "kept_rows": int(keep.sum()),
    "quarantined_rows": int((~keep).sum()),
    "steps_per_epoch": steps,
    "total_steps": int(epochs) * steps,
    "batch_size": int(batch),
    "sources": collection.get("sources"),
    "sampling": collection.get("sampling"),
    "weight_mass_by_source": mass,
    "component_dataset_digests": collection.get("component_dataset_digests"),
    "dagger_component": {k: dagger[0].get(k) for k in ("source_kind", "prefix", "rows", "dataset_digest", "source_run", "world_seeds", "quarantined_rows")},
    "dagger_tagged_kept_rows": tagged_dagger,
    "family_weighting": record,
    "load_seconds": round(seconds, 1),
    "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 2),
}, open(out, "w"), indent=1, sort_keys=True)
print(json.dumps({"rows": n, "steps_per_epoch": steps, "dagger_tagged_kept_rows": tagged_dagger,
                  "peak_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 2)}))
PY
    mv "$out.tmp" "$out"
  fi
  local spe
  spe=$("$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["steps_per_epoch"], d["rows"], d["peak_rss_gib"], d["dagger_tagged_kept_rows"])' "$out")
  read -r DC_SPE DC_ROWS DC_RSS DC_TAGGED <<< "$spe"
  [ "$DC_ROWS" -eq "$N_ROWS" ] || die "datacheck rows $DC_ROWS != N $N_ROWS"
  log "datacheck: N=$DC_ROWS steps/epoch=$DC_SPE, DAgger family-tagged kept rows=$DC_TAGGED, load peak RSS ${DC_RSS} GiB"
}

write_config() {  # write_config start|final -> config.json (local; uploaded once at the end) and config-<launch>.json
  PHASE=$1 RUN_ID="$RUN_ID" LAUNCH="$LAUNCH" SHA="$SHA" BALANCE_ARGS="$BALANCE_ARGS" R1_BALANCE_ARGS="$R1_BALANCE_ARGS" \
  DEST="$DEST" R1_GCS="$R1_GCS" R1_CODE="$R1_CODE" R1_ACTOR_SHA256="$R1_ACTOR_SHA256" R1_S_PER_STEP="$R1_S_PER_STEP" \
  CODE_COMMIT="$(cat "$CODEROOT/COMMIT" 2> /dev/null)" TARBALL_SHA256="$(cat "$CODEROOT/TARBALL_SHA256" 2> /dev/null)" CODE_GCS="$CODE_GCS" \
  INIT_GCS="$INIT_GCS" INIT_SHA256="${INIT_SHA256_ACTUAL:-}" ACTOR_NAME="$ACTOR_NAME" \
  TEACHER_ROWS="${TEACHER_ROWS:-}" DAGGER_ROWS="${DAGGER_ROWS:-}" ROWS="${N_ROWS:-}" SPE="${SPE:-}" TOTAL_STEPS="${TOTAL:-}" \
  BC_BATCHES="$BC_BATCHES" FIDELITY_BATCHES="$FIDELITY_BATCHES" EPOCHS="$EPOCHS" SEED="$SEED" LR="$LR" BATCH="$BATCH" \
  DEVICE="$DEVICE" FAMILY_WEIGHTS="$FAMILY_WEIGHTS" BC_CMD="${BC_CMD:-}" WORK="$WORK" BC_TIME_FACTOR="$BC_TIME_FACTOR" \
  DAGGER_GCS="$DAGGER_GCS" DAGGER_STATUS="$DAGGER_STATUS" DAGGER_DIR="$DAGGER_DIR" DAGGER_ACTOR_SHA256="$DAGGER_ACTOR_SHA256" \
  "$PY" - <<'PY'
import json, os, platform, socket, subprocess, time
E = os.environ
work = E["WORK"]
cfg_path = os.path.join(work, "config.json")
cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
def opt_int(key):
    return int(E[key]) if E.get(key) else None
dagger = {"gcs": E["DAGGER_GCS"], "status_object": E["DAGGER_STATUS"], "local": E["DAGGER_DIR"],
          "collector_actor_sha256": E["DAGGER_ACTOR_SHA256"], "rows": opt_int("DAGGER_ROWS")}
manifest = os.path.join(E["DAGGER_DIR"], "dataset.json")
if os.path.exists(manifest):
    m = json.load(open(manifest))
    o = m["source_record"]["options"]
    dagger.update(dataset_digest=m["dataset_digest"], source_run=m["source_run"],
                  learner_commit=m["source_record"].get("learner_commit"), prefix=o.get("prefix"),
                  actor_mode=o.get("actor_mode"), starts_per_world=o.get("starts_per_world"),
                  depth=o.get("depth"), horizon=o.get("turns"), round=o.get("round"),
                  worlds=sorted(int(w) for w in m["shards"]))
status = os.path.join(work, "DAGGER_STATUS")
if os.path.exists(status):
    dagger["status"] = open(status).read().strip()
datacheck_path = os.path.join(work, "datacheck.json")
datacheck = json.load(open(datacheck_path)) if os.path.exists(datacheck_path) else None
if datacheck is not None:
    dagger["kept_rows"] = (datacheck.get("sources") or {}).get("learner_state_dagger", {}).get("kept_rows")
    dagger["family_tagged_kept_rows"] = datacheck.get("dagger_tagged_kept_rows")
cfg.update(
    schema="r5-actor-refit-dagger-config-v1",
    run_id=E["RUN_ID"],
    intended_difference=(
        "R1's pretrain-bc command with the actor-from-reset DAgger dataset appended as one more "
        "--dataset (pretrain-bc gives the four teacher batches one source, row_source 0, and the "
        "DAgger rows the other, row_source 1: equal source mass, equal mass per world within each "
        "source; family weights are rescaled inside each (source, world)), the LR schedule length "
        "recomputed from the combined row count, and actor-from-reset DAgger rows family-tagged by "
        "the teacher-trajectory tracker (labor_scenarios.row_family_tags, one tracker per rollout; "
        "code differs from R1's df1aac5 by design). Everything else equals R1 (same_as_r1)."
    ),
    code={"sha": E["SHA"], "archived_commit": E["CODE_COMMIT"], "tarball_sha256": E["TARBALL_SHA256"], "tarball": E["CODE_GCS"]},
    balance_args=E["BALANCE_ARGS"].split(),
    r1_reference={
        "gcs": E["R1_GCS"], "code": E["R1_CODE"], "actor_sha256": E["R1_ACTOR_SHA256"],
        "s_per_step": float(E["R1_S_PER_STEP"]), "rows": 136472, "steps_per_epoch": 534, "total_steps": 13350,
        "balance_args": E["R1_BALANCE_ARGS"].split(),
    },
    same_as_r1={
        "bc_batches": E["BC_BATCHES"].split(), "epochs": int(E["EPOCHS"]), "seed": int(E["SEED"]),
        "lr": E["LR"], "batch_size": int(E["BATCH"]), "device": E["DEVICE"],
        "family_weights": E["FAMILY_WEIGHTS"], "family_weight_decay": "cosine",
        "lr_schedule": "warmup-cosine warmup=0.1 final=0.1", "max_grad_norm": 1.0,
        "stability": "--stability-early-stop", "allow_vm_scale": True,
        "balance_args_equal_r1": E["BALANCE_ARGS"].split() == E["R1_BALANCE_ARGS"].split(),
        "fidelity_batches": E["FIDELITY_BATCHES"].split(),
        "init_checkpoint": {"sha256": E["INIT_SHA256"], "source": E["INIT_GCS"] + "/checkpoint-000.pt"},
    },
    dagger_dataset=dagger,
    rows={"teacher": opt_int("TEACHER_ROWS"), "dagger": opt_int("DAGGER_ROWS"), "total_N": opt_int("ROWS"),
          "definition": "every row pretrain-bc iterates (target_specialty rows, quarantined included at weight 0)"},
    steps_per_epoch=opt_int("SPE"),
    total_steps=opt_int("TOTAL_STEPS"),
    bc_command=E["BC_CMD"] or cfg.get("bc_command"),
    actor=E["ACTOR_NAME"],
    host={"hostname": socket.gethostname(), "python": platform.python_version()},
    dest=E["DEST"],
)
if datacheck is not None:
    cfg["datacheck"] = {k: datacheck.get(k) for k in ("rows", "kept_rows", "quarantined_rows", "steps_per_epoch", "total_steps",
                                                       "sources", "sampling", "weight_mass_by_source", "load_seconds", "peak_rss_gib")}
    # Shares at the start of the fit (full up-weights; the cosine decay takes them to 1), overall
    # and per source ("0" teacher, "1" DAgger), plus what is tagged.
    cfg["family_weighting"] = datacheck.get("family_weighting")
cfg.setdefault("started_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
events = os.path.join(work, "resume_events.jsonl")
cfg["resume_events"] = [json.loads(l) for l in open(events)] if os.path.exists(events) else []
cfg["uninterrupted"] = not cfg["resume_events"]
cfg.setdefault("launches", [])
if E["LAUNCH"] not in cfg["launches"]:
    cfg["launches"].append(E["LAUNCH"])
if E["PHASE"] == "final":
    cfg["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    timings = os.path.join(work, "timings.tsv")
    if os.path.exists(timings):
        cfg["stage_seconds"] = {l.split("\t")[0]: int(l.split("\t")[2]) for l in open(timings) if l.strip()}
    sha = os.path.join(work, E["ACTOR_NAME"] + ".sha256")
    if os.path.exists(sha):
        cfg["actor_sha256"] = open(sha).read().split()[0]
    bc = os.path.join(work, "bc.log")
    if os.path.exists(bc):
        import re
        vals = re.findall(r"s_per_step=([0-9.]+)", open(bc, errors="replace").read())
        if vals:
            cfg["bc_s_per_step_final"] = float(vals[-1])
            cfg["bc_step_time_ratio_vs_r1"] = round(float(vals[-1]) / float(E["R1_S_PER_STEP"]), 3)
    schedule = os.path.join(work, "bc-schedule.json")
    if os.path.exists(schedule):
        cfg["bc_schedule_result"] = json.load(open(schedule))
    try:
        cfg["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        pass
tmp = cfg_path + ".tmp"
json.dump(cfg, open(tmp, "w"), indent=1, sort_keys=True)
os.replace(tmp, cfg_path)
with open(os.path.join(work, "config-" + E["LAUNCH"] + ".json"), "w") as f:
    json.dump(cfg, f, indent=1, sort_keys=True)
PY
}

stage_bc() {
  actor="$OUTD/$ACTOR_NAME"
  local bc_dirs=() b
  for b in $BC_BATCHES; do bc_dirs+=("$DATA/$b"); done
  bc_dirs+=("$DAGGER_DIR")   # the only added --dataset
  # N = teacher + DAgger shard rows = the loaded target_specialty rows (datacheck); pretrain-bc's
  # steps per epoch are ceil(N / BATCH) (labor_bc.pretrain_labor_actor_bc).
  ROWS=$N_ROWS
  SPE=$(( (ROWS + BATCH - 1) / BATCH ))
  TOTAL=$(( EPOCHS * SPE ))
  [ "$SPE" -eq "$DC_SPE" ] || die "steps/epoch $SPE != datacheck $DC_SPE"
  local eta_s
  eta_s=$(awk -v n="$TOTAL" -v s="$R1_S_PER_STEP" -v f="$BC_TIME_FACTOR" 'BEGIN{printf "%d", n*s*f}')
  log "BC rows N=$ROWS (teacher $TEACHER_ROWS + DAgger $DAGGER_ROWS) steps/epoch=$SPE total_steps=$TOTAL (R1: 136472 / 534 / 13350); ETA ~$(( eta_s / 60 )) min at ${R1_S_PER_STEP} s/step x factor ${BC_TIME_FACTOR}"
  write_config start
  if [ -f "$actor" ]; then log "BC output present locally; skipping BC"; return 0; fi
  if remote_has "$DEST/$ACTOR_NAME" && remote_has "$DEST/$ACTOR_NAME.sha256"; then
    log "BC output present in $DEST; fetching instead of retraining"
    timeout 600 gcloud storage cp --no-user-output-enabled "$DEST/$ACTOR_NAME" "$actor" < /dev/null \
      && timeout 600 gcloud storage cp --no-user-output-enabled "$DEST/$ACTOR_NAME.sha256" "$actor.sha256" < /dev/null \
      || die "fetch of the finished actor failed"
    [ "$(sha256_of "$actor")" = "$(cut -d' ' -f1 "$actor.sha256")" ] || die "fetched actor sha256 mismatch"
    return 0
  fi
  local ck=$init epochs=$EPOCHS start=0 args=() d
  if [ -f "$actor.partial" ]; then
    # R1's resume path, verbatim.
    start=$("$PY" -c 'import torch,sys; d=torch.load(sys.argv[1],map_location="cpu",weights_only=False); print(int((d.get("g0_partial") or {}).get("steps",0)))' "$actor.partial")
    cp "$actor.partial" "$OUTD/actor.resume.pt"; ck=$OUTD/actor.resume.pt
    epochs=$(( EPOCHS - start / SPE )); [ "$epochs" -lt 1 ] && epochs=1
    log "resume BC from partial at step $start of $TOTAL ($epochs epochs left; Adam, shuffle, family decay and GradNorm state restart)"
    printf '{"utc":"%s","partial_steps":%s,"total_steps":%s}\n' "$(ts)" "$start" "$TOTAL" >> "$WORK/resume_events.jsonl"
  fi
  for d in "${bc_dirs[@]}"; do args+=(--dataset "$d"); done
  # R1's command (bc-command.txt of actor-refit-r1), with the DAgger --dataset as the 5th dataset.
  local cmd=("$PY" "$CODE/tools/g0launch.py" pretrain-bc --checkpoint "$ck" "${args[@]}"
    --epochs "$epochs" --batch-size "$BATCH" --lr "$LR" --lr-schedule warmup-cosine
    --lr-warmup-fraction 0.1 --lr-final-fraction 0.1
    --lr-schedule-total-steps "$TOTAL" --lr-schedule-start-step "$start" --max-grad-norm 1.0
    --family-weights "$FAMILY_WEIGHTS" --family-weight-decay cosine
    --stability-log "$OUTD/stability-bc.jsonl" --stability-early-stop
    --device "$DEVICE" --allow-vm-scale "${balance_args[@]}" --out "$actor")
  BC_CMD="${cmd[*]}"
  echo "$BC_CMD" > "$OUTD/bc-command.txt"
  write_config start
  gcs_put "$WORK/config-$LAUNCH.json" "$DEST/config-$LAUNCH.json"   # fresh name per launch
  local floor=21600 limit
  limit=$(awk -v e="$eta_s" -v fl="$floor" 'BEGIN{x=3*e+1800; print (x>fl?x:fl)}')
  limit="${BC_TIMEOUT:-$limit}"
  log "pretrain-bc (timeout ${limit}s)"
  (cd "$CODE" && timeout "$limit" "${cmd[@]}" >> "$OUTD/bc.log" 2>&1 < /dev/null) || die "pretrain-bc (see bc.log)"
  rm -f "$OUTD/actor.resume.pt"
  [ -f "$actor" ] || die "pretrain-bc exited 0 but wrote no $ACTOR_NAME"
  (cd "$OUTD" && sha256sum "$ACTOR_NAME" > "$ACTOR_NAME.sha256")
  log "BC done: $(grep -o 's_per_step=[0-9.]*' "$OUTD/bc.log" | tail -1) (R1 ${R1_S_PER_STEP})"
}

stage_loadcheck() {
  [ -f "$actor.sha256" ] || (cd "$OUTD" && sha256sum "$ACTOR_NAME" > "$ACTOR_NAME.sha256")
  "$PY" -c '
import sys, torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
print("weights_only load ok; keys:", sorted(state)[:12])
' "$actor" >> "$LOG_FILE" 2>&1 || die "the actor does not load with weights_only=True (the panel evaluator requires that)"
  # The LR schedule must span exactly EPOCHS x ceil(N / BATCH) steps (the last step at the end
  # of the cosine); an early stop ends sooner and is recorded, not an error.
  "$PY" - "$actor" "$TOTAL" "$OUTD/bc-schedule.json" <<'PY' >> "$LOG_FILE" 2>&1 || die "the BC checkpoint's LR schedule does not span the planned steps (see driver.log)"
import json, sys, torch
actor, total, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
summary = torch.load(actor, map_location="cpu", weights_only=True)["heuristic_bc_pretraining"]
schedule = summary.get("lr_schedule") or {}
fw = summary.get("family_weighting") or {}
record = {"total_steps": schedule.get("total_steps"), "start_step": schedule.get("start_step"),
          "steps_this_fit": schedule.get("steps_this_fit"), "end_step": schedule.get("end_step"),
          "first_lr": schedule.get("first_lr"), "last_lr": schedule.get("last_lr"),
          "planned_total_steps": total, "ended_at_planned_end": schedule.get("end_step") == total,
          "family_weight_schedule": summary.get("family_weight_schedule"),
          "family_weighting_tagged_sources": fw.get("tagged_sources")}
json.dump(record, open(out, "w"), indent=1, sort_keys=True)
print(json.dumps(record))
assert int(schedule["total_steps"]) == total, (schedule.get("total_steps"), total)
assert int(schedule["end_step"]) <= total, (schedule.get("end_step"), total)
PY
  local ended
  ended=$("$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["end_step"], d["ended_at_planned_end"])' "$OUTD/bc-schedule.json")
  case "$ended" in *True) log "LR schedule ended exactly at the planned step $TOTAL" ;;
    *) log "WARN BC ended at step ${ended% *} of $TOTAL (early stop or resumed fit; see stability-bc.jsonl / resume_events)" ;; esac
}

stage_fidelity() {
  local b out t_fid rc=0
  for b in $FIDELITY_BATCHES; do
    out="$OUTD/fidelity-$b.json"
    [ -f "$out" ] && continue
    if remote_has "$DEST/fidelity-$b.json"; then log "fidelity $b already in $DEST"; continue; fi
    [ -f "$DATA/$b/dataset.json" ] || { log "skip fidelity on $b (not staged)"; rc=1; continue; }
    log "actor-fidelity $b"
    t_fid=$(date +%s)
    # R1's fidelity command, verbatim.
    (cd "$CODE" && "$PY" -m $M.labor_pretrain_eval actor-fidelity --checkpoint "$actor" --dataset "$DATA/$b" \
      --batch-size 256 --out "$out" > "$OUTD/fidelity-$b.log" 2>&1 < /dev/null) \
      || { log "actor-fidelity $b FAILED (see fidelity-$b.log)"; rc=1; }
    log "actor-fidelity $b took $(( $(date +%s) - t_fid )) s"
  done
  return "$rc"
}

stage_upload() {
  write_config final
  log "upload to $DEST (each object once)"
  gcs_put_once "$actor" "$DEST/$ACTOR_NAME" || die "actor upload failed"
  gcs_put_once "$actor.sha256" "$DEST/$ACTOR_NAME.sha256" || die "actor sha256 upload failed"
  upload_results || die "a result upload failed (rerun the same line: present objects are skipped)"
  gcs_put_once "$WORK/config.json" "$DEST/config.json" || die "config.json upload failed"
  remote_has "$DEST/$ACTOR_NAME" && remote_has "$DEST/config.json" || die "actor/config.json not found in $DEST after upload"
}

# --- main ---------------------------------------------------------------------------------
log "start launch=$LAUNCH sha=$SHA run_id=$RUN_ID device=$DEVICE epochs=$EPOCHS seed=$SEED lr=$LR batches=[$BC_BATCHES] + dagger=$DAGGER_GCS balance=[$BALANCE_ARGS] dest=$DEST"
rm -f "$WORK/EXIT_STATUS"
timed preflight stage_preflight
timed code stage_code
# Background progress upload (small files, fresh names: objects are write-once) so a stalled
# run can be inspected from GCS.
(
  while sleep "${UPLOAD_EVERY:-900}"; do
    stamp=$(date -u +%H%M%SZ)
    [ -f "$LAUNCH_LOG" ] && gcs_put "$LAUNCH_LOG" "$DEST/progress/$LAUNCH/$stamp-driver.log"
    [ -f "$WORK/bc.log" ] && tail -n 200 "$WORK/bc.log" > "$WORK/logs/bc-tail.log" \
      && gcs_put "$WORK/logs/bc-tail.log" "$DEST/progress/$LAUNCH/$stamp-bc-tail.log"
  done
) > /dev/null 2>&1 < /dev/null &
uploader_pid=$!
timed data stage_data
timed init stage_init
timed datacheck stage_datacheck
timed bc stage_bc
timed loadcheck stage_loadcheck
fid_rc=0
timed_soft fidelity stage_fidelity || fid_rc=$?
timed upload stage_upload
if [ "$fid_rc" -ne 0 ]; then CUR_STAGE=fidelity; die "actor-fidelity failed or skipped for some batch (the actor and the other outputs were uploaded)"; fi
CUR_STAGE="done"
echo "0 $(ts) rows=$N_ROWS teacher=$TEACHER_ROWS dagger=$DAGGER_ROWS steps=$TOTAL actor=$(cut -c1-16 "$actor.sha256") sha=$SHA" > "$WORK/EXIT_STATUS"
log "done: $actor ($(cut -c1-16 "$actor.sha256")); EXIT_STATUS uploaded last by the exit handler"
