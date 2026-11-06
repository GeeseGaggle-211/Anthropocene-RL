#!/usr/bin/env bash
# r5 actor DAgger collection (CPU VM): the actor's own 50-decision episodes from the reset on
# the fitting worlds, every visited state labelled by the H12 teacher.
#
# Derived from actor_diag_job.sh (code/bridge/checkpoint staging, sha checks, duplicate guards,
# write-once GCS status, logging). One collect process:
#   labor_dagger collect --prefix actor-from-reset --depth 50 --horizon 50 --starts 1 --round 1
#     --players 16 --width 128 --height 64 --separation 9 --jobs $SLOTS --resume --out $WORK/data
# with the actor checkpoint (default: the R1 refit, the panel-r1 / actor-diag checkpoint) and the
# panel bridge. MODE=stochastic adds --stochastic and STARTS independent rollouts per world
# (each from its own generator keyed by policy seed, world, round and start index).
#
# Stages:
#   probe   (SKIP_PROBE=1 skips; remembered per code+bridge+checkpoint in $WORK/PROBE_OK): on
#           PROBE_WORLD (fitting, default 90100) for PROBE_DECISIONS (5) decisions, a
#           deterministic actor-from-reset collection and labor_pretrain_eval autonomous
#           --modes deterministic (--allow-non-development: the evaluator's existing switch for
#           non-development worlds; sealed worlds stay refused) must give identical per-decision
#           turn records and reset identities (labor_dagger rollout-compare). That proves the
#           teacher's shadow query (capture/reset of the shadow bridge) leaves the actor's
#           closed-loop state untouched, so the collected states are the actor's real ones.
#           The probe dataset stays local (never training data); its comparison is uploaded.
#   collect labor_dagger collect ... --resume. A spot preemption stops the VM but keeps the
#           disk: rerunning the same line resumes from the verified local shards (the source
#           record must match: same code snapshot, bridge, checkpoint and options).
#   upload  after dataset.json is published locally and verifies: every shard file to
#           $DEST/data/ with --no-clobber (each object written once), source.json, then
#           dataset.json last; then $DEST/summary.json (rows, worlds, rows per world, wall
#           times, per-world rollout summaries) and EXIT_STATUS.
#
# GCS status objects. The VM service account cannot overwrite or delete objects, so:
#   * $DEST/EXIT_STATUS is uploaded exactly once, at the end ("0 <utc> ..." or
#     "FAIL:<stage> <utc> ..."). If it already exists at start, the job logs and exits 0 without
#     doing anything: a FAIL finalises this DEST (relaunch with a new DEST).
#   * An interrupted launch (TERM/INT/HUP, e.g. a VM stop) uploads no EXIT_STATUS, so the same
#     line can be rerun: the collector resumes, present remote objects are skipped.
#   * $DEST/RUNNING is the start marker (a failed re-upload on relaunch is harmless and logged).
#   * Objects a relaunch could upload again go to fresh names: logs/job-<launch>.log,
#     logs/collect-<launch>.log, logs/probe-<launch>.{log,json}, config-<launch>.json.
#     CHECKPOINT_SHA256 is uploaded only when absent (a different one refuses this DEST);
#     data/* and summary.json are uploaded only when absent.
#
# Shares the CPU VM: never stops it, nice 10, SLOTS collector processes, a memory floor before
# the collect starts, no crontab or other persistence.
#
# Required env:  SHA (commit of the code tarball)  CODE_GCS (gs:// path of the code tarball)
# Optional env:  ACTOR_GCS=gs://.../r5/scaleout/actor-refit-r1  ACTOR_NAME=actor-bc-refit-r1.pt
#   WORLDS="<the 32 R1 teacher worlds below>" (space list; ranges a-b allowed; fitting worlds
#   90100-90153 only)  MODE=deterministic|stochastic  STARTS=1  POLICY_SEED=0  SLOTS=4
#   MIN_MEM_GB=10  MEM_PER_SLOT_GB=6 (collect waits for MIN_MEM_GB + SLOTS*MEM_PER_SLOT_GB)
#   MEM_WAIT_H=6  MIN_DISK_GB=40  COLLECT_TIMEOUT=96h  PROBE_WORLD=90100  PROBE_DECISIONS=5
#   SKIP_PROBE=1  ALLOW_NO_STATUS=1  PY=... BRIDGE=... BRIDGE_SHA256=... WORK=... DEST=...
#
# Launch (detached; rerun the same line after an interruption):
#   SHA=<sha> CODE_GCS=<tarball> \
#     setsid nohup bash $HOME/actor_dagger_job.sh > $HOME/r5/actor-dagger.out 2>&1 < /dev/null &
set -euo pipefail

SHA="${SHA:?set SHA to the code commit of the CODE_GCS tarball}"
CODE_GCS="${CODE_GCS:?set CODE_GCS to the gs:// path of the code tarball (git archive of SHA)}"
GCS_ROOT="${GCS_ROOT:-gs://anthropocene-rl-datasets-508912/model9/r5}"
ACTOR_GCS="${ACTOR_GCS:-$GCS_ROOT/scaleout/actor-refit-r1}"
ACTOR_NAME="${ACTOR_NAME:-actor-bc-refit-r1.pt}"
DEST="${DEST:-$GCS_ROOT/scaleout/actor-dagger-r1}"
WORK="${WORK:-$HOME/r5/actor-dagger}"
CODEROOT="${CODEROOT:-$HOME/code/rl-dagger-$SHA}"
CODE="$CODEROOT/rl"
PY="${PY:-$HOME/m9-bench/venv/bin/python}"
BRIDGE="${BRIDGE:-$HOME/code/r5s841c38e/rl/integration/bin/model9/Anthropocene.Model9Bridge.dll}"
BRIDGE_SHA256="${BRIDGE_SHA256:-99131dce9c7f0d0d66988b05c3c3e9adea6f6ad3407ce4cd0366d6a14b25eb6b}"
# The 32 fitting worlds of the teacher batches R1 was trained on: the shard keys of
# $GCS_ROOT/pilot/{fit8-b01,fit8-b02,fit8-b03r,fit8-b04}/dataset.json (fit8-b03r replaces
# fit8-b03; R1's bc-command.txt lists exactly these four --dataset batches) = 90100-90131.
# None lies in the reserved held-out range 90154-90159.
WORLDS="${WORLDS:-90100 90101 90102 90103 90104 90105 90106 90107 90108 90109 90110 90111 90112 90113 90114 90115 90116 90117 90118 90119 90120 90121 90122 90123 90124 90125 90126 90127 90128 90129 90130 90131}"
MODE="${MODE:-deterministic}"
STARTS="${STARTS:-1}"
POLICY_SEED="${POLICY_SEED:-0}"
SLOTS="${SLOTS:-4}"
MIN_MEM_GB="${MIN_MEM_GB:-10}"
MEM_PER_SLOT_GB="${MEM_PER_SLOT_GB:-6}"
MEM_WAIT_H="${MEM_WAIT_H:-6}"
MIN_DISK_GB="${MIN_DISK_GB:-40}"
COLLECT_TIMEOUT="${COLLECT_TIMEOUT:-96h}"
PROBE_WORLD="${PROBE_WORLD:-90100}"
PROBE_DECISIONS="${PROBE_DECISIONS:-5}"
# The panel's parameters (cpu_sched.py defaults); one aggregation round from the reset.
TURNS=50; PLAYERS=16; WIDTH=128; HEIGHT=64; SEPARATION=9; ROUND=1

ts() { date -u +%FT%TZ; }
LAUNCH=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$WORK/logs" "$WORK/probe" "$WORK/ckpt"
LOG_FILE="$WORK/logs/job-$LAUNCH.log"
CUR_STAGE=startup
log() { echo "$(ts) [actor-dagger] $*" | tee -a "$LOG_FILE" >&2; }
die() { log "FAIL[$CUR_STAGE] $*"; exit 1; }
sha256_of() { sha256sum "$1" | cut -d' ' -f1; }
gcs_put() { timeout 600 gcloud storage cp --no-user-output-enabled "$1" "$2" < /dev/null || log "upload failed: $1 -> $2"; }
gcs_put_once() { remote_has "$2" && { log "present, not re-uploaded: $2"; return 0; }; timeout 600 gcloud storage cp --no-user-output-enabled "$1" "$2" < /dev/null; }
remote_has() { timeout 120 gcloud storage ls "$1" > /dev/null 2>&1 < /dev/null; }
mem_available_gb() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
disk_free_gb() { df -Pk "$1" | awk 'NR==2{print int($4/1048576)}'; }

# Duplicate guards (before the EXIT trap: a refused duplicate must not write EXIT_STATUS). The
# [x] brackets keep pgrep from matching its own command line.
exec 9> "$WORK/job.lock"
flock -n 9 || { log "another actor_dagger_job.sh holds $WORK/job.lock; exiting"; exit 3; }
if pgrep -f "[l]abor_dagger collect .*$WORK/|[l]abor_pretrain_eval autonomous .*$WORK/probe/" > /dev/null; then
  log "actor-dagger workers are already running: $(pgrep -af "[l]abor_dagger collect .*$WORK/" | head -3); exiting"
  exit 3
fi
# A final EXIT_STATUS cannot be replaced (no delete permission): this DEST is finished.
if remote_has "$DEST/EXIT_STATUS"; then
  log "$DEST/EXIT_STATUS already exists ($(timeout 120 gcloud storage cat "$DEST/EXIT_STATUS" < /dev/null 2> /dev/null | head -1)); nothing to do (use a new DEST to rerun)"
  exit 0
fi

SKIP_STATUS=0   # 1: this launch must not upload EXIT_STATUS (interrupted, or DEST refused)
finish() {
  local status=$?
  trap - EXIT
  set +e
  if [ "$SKIP_STATUS" != "1" ] && [ "$status" -ne 0 ] && [ ! -f "$WORK/EXIT_STATUS" ]; then
    echo "FAIL:$CUR_STAGE $(ts)" > "$WORK/EXIT_STATUS"
  fi
  gcs_put "$LOG_FILE" "$DEST/logs/job-$LAUNCH.log"
  if [ "$SKIP_STATUS" != "1" ] && [ -f "$WORK/EXIT_STATUS" ]; then
    if remote_has "$DEST/EXIT_STATUS"; then
      log "$DEST/EXIT_STATUS appeared during this launch; not overwriting (local: $(cat "$WORK/EXIT_STATUS"))"
    else
      gcs_put "$WORK/EXIT_STATUS" "$DEST/EXIT_STATUS"   # last and once; the VM is never stopped by this job
    fi
  fi
  exit "$status"
}
trap finish EXIT
trap 'SKIP_STATUS=1; log "signal received: interrupted, no EXIT_STATUS (rerun the same line to resume)"; exit 143' TERM INT HUP

# --- configuration checks ------------------------------------------------------------------
worlds=()
for part in $WORLDS; do
  case "$part" in
    *-*) for ((w = ${part%-*}; w <= ${part#*-}; w++)); do worlds+=("$w"); done ;;
    *) worlds+=("$part") ;;
  esac
done
[ "${#worlds[@]}" -gt 0 ] || die "WORLDS is empty"
fitting_world() {  # fitting worlds only: never sealed 3000-3059, development 2000-2059 or reserved 90154-90159
  case "$1" in ''|*[!0-9]*) return 1 ;; esac
  [ "$1" -ge 90100 ] && [ "$1" -le 90153 ]
}
for w in "${worlds[@]}"; do
  fitting_world "$w" || die "world $w is not a fitting world 90100-90153 (90154-90159 are reserved held-out, 3000-3059 sealed)"
done
[ "$(printf '%s\n' "${worlds[@]}" | sort -u | wc -l)" -eq "${#worlds[@]}" ] || die "WORLDS repeats a world"
fitting_world "$PROBE_WORLD" || die "PROBE_WORLD $PROBE_WORLD is not a fitting world 90100-90153"
case "$MODE" in
  deterministic) [ "$STARTS" = "1" ] || die "MODE=deterministic repeats one rollout: STARTS must be 1" ; MODE_ARGS=() ;;
  stochastic) MODE_ARGS=(--stochastic) ;;
  *) die "MODE must be deterministic or stochastic, not '$MODE'" ;;
esac
for v in STARTS POLICY_SEED SLOTS MIN_MEM_GB MEM_PER_SLOT_GB MIN_DISK_GB PROBE_DECISIONS; do
  case "${!v}" in ''|*[!0-9]*) die "$v must be a non-negative integer, not '${!v}'" ;; esac
done
[ "$STARTS" -ge 1 ] && [ "$SLOTS" -ge 1 ] && [ "$PROBE_DECISIONS" -ge 1 ] || die "STARTS, SLOTS and PROBE_DECISIONS must be positive"
WORLD_LIST=$(IFS=,; echo "${worlds[*]}")

# --- stages ---------------------------------------------------------------------------------
tar_commit_id() {  # TARBALL -> git archive commit id (empty when none); SIGPIPE-safe under pipefail
  local id=""
  if command -v git > /dev/null 2>&1; then id=$(gzip -dc "$1" | git get-tar-commit-id 2> /dev/null || true); fi
  if [ -z "$id" ]; then
    id=$("$PY" -c 'import sys,tarfile; t=tarfile.open(sys.argv[1]); t.next(); print(t.pax_headers.get("comment",""))' "$1" 2> /dev/null || true)
  fi
  echo "$id"
}

stage_code() {
  CUR_STAGE=code
  [ -x "$PY" ] || die "missing venv python $PY"
  command -v git > /dev/null 2>&1 || die "git is required (the collector records a clean learner commit)"
  if [ ! -f "$CODEROOT/.snapshot-ok" ]; then
    log "fetching $CODE_GCS"
    rm -rf "$CODEROOT" "$CODEROOT.tmp"
    mkdir -p "$CODEROOT.tmp/rl"
    timeout 600 gcloud storage cp --no-user-output-enabled "$CODE_GCS" "$CODEROOT.tmp/code.tar.gz" < /dev/null \
      || die "download failed: $CODE_GCS (build and upload the tarball first)"
    local archived
    archived=$(tar_commit_id "$CODEROOT.tmp/code.tar.gz")
    [ -n "$archived" ] || die "tarball has no commit id"
    [ "$archived" = "$SHA" ] || [ "${archived:0:${#SHA}}" = "$SHA" ] || die "tarball commit $archived != SHA $SHA"
    tar -xzf "$CODEROOT.tmp/code.tar.gz" -C "$CODEROOT.tmp/rl"
    echo "$archived" > "$CODEROOT.tmp/COMMIT"
    # The production source record needs a clean 40-hex learner commit (provenance.code_commit):
    # a local snapshot commit of the archive with fixed identity and dates, so the same tarball
    # always gives the same snapshot id. Required here (not best-effort): --resume binds to it.
    (cd "$CODEROOT.tmp/rl" && git init -q && git add -A -f \
      && GIT_AUTHOR_DATE="2000-01-01T00:00:00Z" GIT_COMMITTER_DATE="2000-01-01T00:00:00Z" \
         git -c user.name=dagger -c user.email=dagger@localhost -c commit.gpgsign=false \
         commit -q -m "snapshot of $archived (git archive)") > /dev/null 2>&1 \
      || die "snapshot commit failed"
    mv "$CODEROOT.tmp" "$CODEROOT"
    touch "$CODEROOT/.snapshot-ok"
  fi
  SOURCE_COMMIT=$(cat "$CODEROOT/COMMIT")
  SNAPSHOT_COMMIT=$(git -C "$CODE" rev-parse HEAD)
  [ -z "$(git -C "$CODE" status --porcelain --untracked-files=no 2> /dev/null)" ] || die "code snapshot $CODE is dirty"
  export PYTHONPATH="$CODE/src" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  export ANTHROPOCENE_DOTNET="${ANTHROPOCENE_DOTNET:-/usr/bin/dotnet}"
  export ANTHROPOCENE_MODEL9_BRIDGE="$BRIDGE"
  unset ANTHROPOCENE_CODE_COMMIT   # the snapshot's own HEAD is the recorded learner commit
  [ -f "$BRIDGE" ] || die "missing bridge $BRIDGE"
  [ "$(sha256_of "$BRIDGE")" = "$BRIDGE_SHA256" ] || die "bridge sha256 differs from the panel bridge (set BRIDGE_SHA256 if the rebuild is intended)"
  [ -x "$ANTHROPOCENE_DOTNET" ] || die "missing dotnet $ANTHROPOCENE_DOTNET"
  local where checked
  where=$(cd "$CODE" && "$PY" -c 'import anthropocene_rl,sys; print(anthropocene_rl.__file__)' 2>&1) || die "cannot import anthropocene_rl: $where"
  case "$where" in "$CODE"/src/*) ;; *) die "anthropocene_rl resolves to $where, not $CODE/src" ;; esac
  # The code must know --prefix actor-from-reset and rollout-compare, and record a clean commit.
  checked=$(cd "$CODE" && "$PY" -c 'import sys
from anthropocene_rl.autonomous_v7 import labor_dagger as D
from anthropocene_rl.autonomous_v7.provenance import code_commit
assert D.PREFIX_ACTOR_FROM_RESET in D.PREFIXES and hasattr(D, "run_rollout_compare")
c = code_commit()
assert len(c) == 40 and all(x in "0123456789abcdef" for x in c), c
print(c)' 2>&1 < /dev/null) || die "code $SOURCE_COMMIT cannot run this job: $checked"
  [ "$checked" = "$SNAPSHOT_COMMIT" ] || die "learner commit $checked != snapshot $SNAPSHOT_COMMIT"
  log "code $SOURCE_COMMIT (snapshot $SNAPSHOT_COMMIT) in $CODE; bridge sha256 $(sha256_of "$BRIDGE" | cut -c1-16); free mem $(mem_available_gb) GB; free disk $(disk_free_gb "$WORK") GB; cpus $(OMP_NUM_THREADS= nproc --all)"
  [ "$(disk_free_gb "$WORK")" -ge "$MIN_DISK_GB" ] || [ -f "$WORK/data/dataset.json" ] || die "free disk $(disk_free_gb "$WORK") GB < MIN_DISK_GB=$MIN_DISK_GB in $WORK"
}

stage_checkpoint() {
  CUR_STAGE=checkpoint
  local status_text first
  status_text=$(timeout 120 gcloud storage cat "$ACTOR_GCS/EXIT_STATUS" < /dev/null 2> /dev/null || true)
  first=${status_text%% *}
  if [ "$first" != "0" ]; then
    if [ -z "$first" ] && [ "${ALLOW_NO_STATUS:-0}" = "1" ] && remote_has "$ACTOR_GCS/$ACTOR_NAME"; then
      log "WARN no EXIT_STATUS in $ACTOR_GCS; ALLOW_NO_STATUS=1 and the checkpoint exists"
    else
      die "the checkpoint job status is '${status_text:-absent}' in $ACTOR_GCS (needs 0; ALLOW_NO_STATUS=1 accepts a present checkpoint without status)"
    fi
  fi
  CKPT="$WORK/ckpt/$ACTOR_NAME"
  # The remote .sha256 is authoritative: a local copy of an older checkpoint of the same name is replaced.
  timeout 300 gcloud storage cp --no-user-output-enabled "$ACTOR_GCS/$ACTOR_NAME.sha256" "$CKPT.sha256.new" < /dev/null || die "cannot fetch $ACTOR_NAME.sha256"
  if [ ! -f "$CKPT" ] || [ "$(sha256_of "$CKPT")" != "$(cut -d' ' -f1 "$CKPT.sha256.new")" ]; then
    rm -f "$CKPT"
    timeout 900 gcloud storage cp --no-user-output-enabled "$ACTOR_GCS/$ACTOR_NAME" "$CKPT" < /dev/null || die "cannot fetch $ACTOR_GCS/$ACTOR_NAME"
  fi
  mv "$CKPT.sha256.new" "$CKPT.sha256"
  CKPT_SHA256=$(sha256_of "$CKPT")
  [ "$CKPT_SHA256" = "$(cut -d' ' -f1 "$CKPT.sha256")" ] || die "checkpoint sha256 differs from $ACTOR_NAME.sha256"
  "$PY" -c 'import sys, torch; torch.load(sys.argv[1], map_location="cpu", weights_only=True); print("weights_only load ok")' "$CKPT" \
    >> "$LOG_FILE" 2>&1 || die "the checkpoint does not load with weights_only=True (what the collector requires)"
  # One checkpoint per DEST: never mix data of different checkpoints. Uploaded only when absent.
  local remote_sha
  remote_sha=$(timeout 120 gcloud storage cat "$DEST/CHECKPOINT_SHA256" < /dev/null 2> /dev/null | cut -d' ' -f1 || true)
  echo "$CKPT_SHA256" > "$WORK/CHECKPOINT_SHA256"
  if [ -n "$remote_sha" ] && [ "$remote_sha" != "$CKPT_SHA256" ]; then
    SKIP_STATUS=1
    die "$DEST already holds data of checkpoint $remote_sha, not $CKPT_SHA256; use another DEST"
  fi
  [ -n "$remote_sha" ] || gcs_put "$WORK/CHECKPOINT_SHA256" "$DEST/CHECKPOINT_SHA256"
  echo "RUNNING $(ts) launch=$LAUNCH source=$SOURCE_COMMIT ckpt=${CKPT_SHA256:0:16}" > "$WORK/RUNNING"
  gcs_put "$WORK/RUNNING" "$DEST/RUNNING"   # start marker; fails harmlessly (logged) when a relaunch finds it
  WORK="$WORK" CKPT_SHA256="$CKPT_SHA256" SOURCE_COMMIT="$SOURCE_COMMIT" SNAPSHOT_COMMIT="$SNAPSHOT_COMMIT" \
  SLOTS="$SLOTS" LAUNCH="$LAUNCH" WORLDS="${worlds[*]}" DEST="$DEST" ACTOR="$ACTOR_GCS/$ACTOR_NAME" \
  BRIDGE_SHA256="$BRIDGE_SHA256" MODE="$MODE" STARTS="$STARTS" POLICY_SEED="$POLICY_SEED" ROUND="$ROUND" \
  TURNS="$TURNS" PLAYERS="$PLAYERS" WIDTH="$WIDTH" HEIGHT="$HEIGHT" SEPARATION="$SEPARATION" \
  PROBE_WORLD="$PROBE_WORLD" PROBE_DECISIONS="$PROBE_DECISIONS" \
  "$PY" - <<'PY'
import json, os
E = os.environ
json.dump({
    "schema": "r5-actor-dagger-run-v1", "launch": E["LAUNCH"], "checkpoint": E["ACTOR"],
    "checkpoint_sha256": E["CKPT_SHA256"], "source_commit": E["SOURCE_COMMIT"],
    "snapshot_commit": E["SNAPSHOT_COMMIT"], "bridge_sha256": E["BRIDGE_SHA256"],
    "slots": int(E["SLOTS"]), "dest": E["DEST"], "worlds": [int(w) for w in E["WORLDS"].split()],
    "collect": {"prefix": "actor-from-reset", "mode": E["MODE"], "starts": int(E["STARTS"]),
                "round": int(E["ROUND"]), "depth": int(E["TURNS"]), "horizon": int(E["TURNS"]),
                "policy_seed": int(E["POLICY_SEED"]),
                **{k.lower(): int(E[k]) for k in ("PLAYERS", "WIDTH", "HEIGHT", "SEPARATION")}},
    "probe": {"world": int(E["PROBE_WORLD"]), "decisions": int(E["PROBE_DECISIONS"])},
}, open(os.path.join(E["WORK"], "config-" + E["LAUNCH"] + ".json"), "w"), indent=1, sort_keys=True)
PY
  gcs_put "$WORK/config-$LAUNCH.json" "$DEST/config-$LAUNCH.json"
  log "checkpoint $ACTOR_NAME sha256 ${CKPT_SHA256:0:16} (checkpoint job status: ${status_text:-n/a})"
}

stage_probe() {
  CUR_STAGE=probe
  [ "${SKIP_PROBE:-0}" != "1" ] || { log "probe skipped (SKIP_PROBE=1)"; return 0; }
  local key="$SOURCE_COMMIT $SNAPSHOT_COMMIT $BRIDGE_SHA256 $CKPT_SHA256 $PROBE_WORLD $PROBE_DECISIONS"
  if [ "$(cat "$WORK/PROBE_OK" 2> /dev/null)" = "$key" ]; then log "probe passed earlier for this code, bridge and checkpoint"; return 0; fi
  local data="$WORK/probe/data-$LAUNCH" auto="$WORK/probe/autonomous-$LAUNCH.json"
  local out="$WORK/probe/probe-$LAUNCH.json" plog="$WORK/logs/probe-$LAUNCH.log"
  local panel=(--players "$PLAYERS" --width "$WIDTH" --height "$HEIGHT" --separation "$SEPARATION" --policy-seed "$POLICY_SEED")
  log "probe: world $PROBE_WORLD, $PROBE_DECISIONS decisions: actor-from-reset collector vs labor_pretrain_eval deterministic"
  local rc=0
  (cd "$CODE" && nice -n 10 timeout 4h "$PY" -m anthropocene_rl.autonomous_v7.labor_dagger collect \
      --checkpoint "$CKPT" --worlds "$PROBE_WORLD" --round "$ROUND" --prefix actor-from-reset \
      --depth "$PROBE_DECISIONS" --horizon "$PROBE_DECISIONS" --starts 1 "${panel[@]}" --jobs 1 \
      --out "$data" >> "$plog" 2>&1 < /dev/null) || rc=$?
  if [ "$rc" -eq 0 ]; then
    (cd "$CODE" && nice -n 10 timeout 4h "$PY" -m anthropocene_rl.autonomous_v7.labor_pretrain_eval autonomous \
        --checkpoint "$CKPT" --worlds "$PROBE_WORLD" --turns "$PROBE_DECISIONS" --modes deterministic \
        --allow-non-development "${panel[@]}" --jobs 2 --out "$auto" >> "$plog" 2>&1 < /dev/null) || rc=$?
  fi
  if [ "$rc" -eq 0 ]; then
    (cd "$CODE" && "$PY" -m anthropocene_rl.autonomous_v7.labor_dagger rollout-compare \
        --dagger "$data" --autonomous "$auto" --decisions "$PROBE_DECISIONS" --out "$out" \
        >> "$plog" 2>&1 < /dev/null) || rc=$?
  fi
  gcs_put "$plog" "$DEST/logs/probe-$LAUNCH.log"
  [ -f "$out" ] && gcs_put "$out" "$DEST/logs/probe-$LAUNCH.json"
  [ -f "$auto" ] && gcs_put "$auto" "$DEST/logs/probe-autonomous-$LAUNCH.json"
  [ "$rc" -eq 0 ] || die "probe failed (rc=$rc): the collector's actor-from-reset states differ from the evaluator's deterministic episode, or a probe step failed (see logs/probe-$LAUNCH.log)"
  echo "$key" > "$WORK/PROBE_OK"
  log "probe ok: identical turn records and reset identity on world $PROBE_WORLD ($PROBE_DECISIONS decisions)"
}

stage_collect() {
  CUR_STAGE=collect
  local need waited=0 rc=0 clog="$WORK/logs/collect-$LAUNCH.log"
  if [ -f "$WORK/data/dataset.json" ]; then log "dataset already published in $WORK/data"; fi
  need=$(( MIN_MEM_GB + SLOTS * MEM_PER_SLOT_GB ))
  while [ "$(mem_available_gb)" -lt "$need" ]; do
    [ "$waited" -eq 0 ] && log "MemAvailable $(mem_available_gb) GB < $need GB (MIN_MEM_GB + SLOTS*MEM_PER_SLOT_GB); waiting"
    [ "$waited" -lt "$(awk -v h="$MEM_WAIT_H" 'BEGIN{printf "%d", h*3600}')" ] || die "memory stayed below $need GB for ${MEM_WAIT_H} h"
    sleep 300; waited=$(( waited + 300 ))
  done
  log "collect: ${#worlds[@]} worlds, mode $MODE, starts $STARTS, jobs $SLOTS (mem $(mem_available_gb) GB, disk $(disk_free_gb "$WORK") GB); log $clog"
  COLLECT_STARTED=$(date +%s)
  (cd "$CODE" && nice -n 10 timeout --kill-after=10m "$COLLECT_TIMEOUT" "$PY" -m anthropocene_rl.autonomous_v7.labor_dagger collect \
      --checkpoint "$CKPT" --worlds "$WORLD_LIST" --round "$ROUND" --prefix actor-from-reset \
      --depth "$TURNS" --horizon "$TURNS" --starts "$STARTS" "${MODE_ARGS[@]}" --policy-seed "$POLICY_SEED" \
      --players "$PLAYERS" --width "$WIDTH" --height "$HEIGHT" --separation "$SEPARATION" \
      --jobs "$SLOTS" --resume --out "$WORK/data" >> "$clog" 2>&1 < /dev/null) || rc=$?
  gcs_put "$clog" "$DEST/logs/collect-$LAUNCH.log"
  [ "$rc" -eq 0 ] || die "collect failed rc=$rc (see logs/collect-$LAUNCH.log); rerun the same line to resume from the local shards"
  [ -f "$WORK/data/dataset.json" ] || die "collect exited 0 without $WORK/data/dataset.json"
  log "collect done after $(( ($(date +%s) - COLLECT_STARTED) / 60 )) min"
}

stage_upload() {
  CUR_STAGE=upload
  local summary="$WORK/summary.json"
  # Verify the published dataset and write the summary (also checks the expected worlds/options).
  (cd "$CODE" && WORLDS="${worlds[*]}" MODE="$MODE" STARTS="$STARTS" CKPT_SHA256="$CKPT_SHA256" TURNS="$TURNS" \
    SOURCE_COMMIT="$SOURCE_COMMIT" DEST="$DEST" LAUNCH="$LAUNCH" "$PY" - "$WORK/data" "$summary" <<'PY'
import json, os, sys
from pathlib import Path
from anthropocene_rl.autonomous_v7.labor_teacher_data import verify_dataset, verify_shard

E = os.environ
data, out = Path(sys.argv[1]), Path(sys.argv[2])
manifest = verify_dataset(data)
options = manifest["source_record"]["options"]
worlds = [int(w) for w in E["WORLDS"].split()]
assert options["worlds"] == worlds, (options["worlds"], worlds)
assert options.get("prefix") == "actor-from-reset", options.get("prefix")
assert options["actor_mode"] == E["MODE"] and int(options["starts_per_world"]) == int(E["STARTS"])
assert options["checkpoint_sha256"] == E["CKPT_SHA256"] and int(options["turns"]) == int(E["TURNS"])
assert sorted(int(k) for k in manifest["shards"]) == sorted(worlds)
per_world = {}
for key, entry in sorted(manifest["shards"].items(), key=lambda kv: int(kv[0])):
    meta = verify_shard(data / entry["path"])
    rollouts = meta.get("rollouts") or []
    per_world[key] = {
        "rows": int(entry["rows"]), "decisions": int(entry["decisions"]),
        "native_resolves": int(entry["native_resolves"]), "wall_seconds": float(entry["wall_seconds"]),
        "quarantined": int(meta.get("counters", {}).get("quarantined", 0)),
        "full_action_label_valid": int(meta.get("counters", {}).get("full_action_label_valid", 0)),
        "surviving_camps": int(meta.get("surviving_camps", 0)),
        "rollouts": rollouts,
    }
walls = [w["wall_seconds"] for w in per_world.values()]
json.dump({
    "schema": "r5-actor-dagger-summary-v1", "dest": E["DEST"], "launch": E["LAUNCH"],
    "source_commit": E["SOURCE_COMMIT"], "learner_commit": manifest["source_record"]["learner_commit"],
    "source_run": manifest["source_run"], "dataset_digest": manifest["dataset_digest"],
    "checkpoint_sha256": E["CKPT_SHA256"], "prefix": options["prefix"], "actor_mode": options["actor_mode"],
    "starts_per_world": options["starts_per_world"], "horizon": options["turns"],
    "worlds": len(per_world), "rows": sum(w["rows"] for w in per_world.values()),
    "rows_per_world": {k: w["rows"] for k, w in per_world.items()},
    "wall_seconds_per_world": {k: w["wall_seconds"] for k, w in per_world.items()},
    "wall_seconds_sum": sum(walls), "wall_seconds_max": max(walls),
    "per_world": per_world,
}, open(out, "w"), indent=1, sort_keys=True)
print(json.dumps({"worlds": len(per_world), "rows": sum(w["rows"] for w in per_world.values())}))
PY
  ) >> "$LOG_FILE" 2>&1 || die "the local dataset does not verify against this job's worlds/options (see the job log)"
  # Shards first (each object once: --no-clobber skips objects a previous launch uploaded),
  # then source.json, then dataset.json last: a remote dataset.json implies complete shards.
  local dirs=() d
  for d in "$WORK"/data/world-*; do [ -d "$d" ] && dirs+=("$d"); done
  [ "${#dirs[@]}" -eq "${#worlds[@]}" ] || die "expected ${#worlds[@]} shard directories in $WORK/data, found ${#dirs[@]}"
  if remote_has "$DEST/data/dataset.json"; then
    log "$DEST/data/dataset.json present: data uploaded by an earlier launch"
  else
    timeout 7200 gcloud storage cp -r --no-clobber --no-user-output-enabled "${dirs[@]}" "$DEST/data/" < /dev/null \
      || die "shard upload failed (rerun the same line: present objects are skipped)"
    gcs_put_once "$WORK/data/source.json" "$DEST/data/source.json" || die "source.json upload failed"
    # Every local file must be present remotely before the manifest goes up.
    local missing
    missing=$(comm -23 \
      <(cd "$WORK/data" && find . -type f ! -name dataset.json ! -name '*.tmp' | sed 's|^\./||' | sort) \
      <(timeout 600 gcloud storage ls -r "$DEST/data/**" < /dev/null 2> /dev/null | sed "s|^$DEST/data/||" | sort) | head -5)
    [ -z "$missing" ] || die "remote $DEST/data lacks local files, e.g. $missing"
    gcs_put_once "$WORK/data/dataset.json" "$DEST/data/dataset.json" || die "dataset.json upload failed"
  fi
  gcs_put_once "$summary" "$DEST/summary.json" || die "summary.json upload failed"
  log "uploaded $DEST/data ($(grep -o '"rows": [0-9]*' "$summary" | tail -1)) and $DEST/summary.json"
}

stage_final() {
  CUR_STAGE=final
  remote_has "$DEST/data/dataset.json" && remote_has "$DEST/summary.json" || die "remote dataset.json/summary.json missing after upload"
  local rows
  rows=$("$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["worlds"], d["rows"])' "$WORK/summary.json")
  echo "0 $(ts) worlds=${rows% *} rows=${rows#* } mode=$MODE starts=$STARTS source=$SOURCE_COMMIT ckpt=${CKPT_SHA256:0:16}" > "$WORK/EXIT_STATUS"
  log "done: $(cat "$WORK/EXIT_STATUS") (EXIT_STATUS uploaded last by the exit handler)"
}

# --- main ----------------------------------------------------------------------------------
log "start launch=$LAUNCH sha=$SHA worlds=${#worlds[@]} mode=$MODE starts=$STARTS slots=$SLOTS dest=$DEST"
rm -f "$WORK/EXIT_STATUS"
stage_code
stage_checkpoint
stage_probe
stage_collect
stage_upload
stage_final
