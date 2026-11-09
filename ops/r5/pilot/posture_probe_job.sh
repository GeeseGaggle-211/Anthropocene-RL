#!/usr/bin/env bash
# r5 PPO pilot: DIAGNOSTIC posture probe (CPU VM anthropocene-trainer-60, c3d-highcpu-60 Spot,
# us-central1-c). From a checkpoint (default the D7 start checkpoint, sha 14e749b0...) it runs
# labor_posture_probe: WORLDS x TURNS (default 16, the next PPO run's rollout length) decision
# turns on the training worlds (16 players,
# 128x64, separation 9, movement) under forced non-labor posture arms (default: baseline;
# trade share 0.05/0.2/0.5; fission Encourage; fission Discourage; share 0.2 + Encourage).
# Each arm overrides the submitted fission/trade posture of EVERY camp after the policy
# sampled it; labor actions stay the policy's (per-world sampling seed shared by all arms).
# It records per turn camps/population, terminated/vanished camps, fission daughters, QoL and
# population rewards, the RAW realised trade (trade set columns 6/7/8 per group) and the labor
# compiler's interventions (rejected first proposals, typed violation reasons, fallbacks,
# feasibility_initial/feasibility_retry/settlement_guard as collect stores them).
#
# The output carries scope "diagnostic-posture-override": train, collect, the anchor producer
# and calibration refuse it. It trains nothing and writes no rollout/anchor/checkpoint.
#
# It shares the VM with the running PPO pilot: modest parallelism (JOBS=7 processes, one per
# (arm, world) unit, each with its own native bridge; THREADS=1 torch thread each), nice 10,
# and it refuses to start below MIN_MEM_GB of MemAvailable. Fresh worlds 91300/91301 (not
# D7/step-check 91000-91005, not the pilot's 91100-91190/91200-91290 blocks; sealed 3000-3059
# and reserved 90154-90159 are refused by the probe and the preflight).
#
# Conventions follow anchor_probe_job.sh: code tarball (git archive, commit-id check), shared
# venv and bridge (sha256 99131dce...), OMP/MKL 1 thread, flock + pgrep duplicate guards, a
# whole-job `timeout` (JOB_TIMEOUT, default 120m: 14 (arm, world) units on 7 processes = 2
# waves of 16 turns at ~54+ s/native step single-threaded, more as camps multiply), launch-stamped DEST
# (gs://.../ppo-pilot/posture-probe/<launch>/), EXIT_STATUS uploaded once and LAST
# ("0 <utc> ..." or "FAIL:<stage> <utc>"). It NEVER stops the VM, touches no crontab and no
# other job's WORK/DEST, and kills only processes it observed in its own probe's tree.
#
# Required env: SHA (the code commit; its tarball rl-pilot-<sha7>.tar.gz uploaded first).
# Optional: GCS_ROOT CODE_GCS CKPT_GCS CKPT_SHA256 DEST WORK CODEROOT PY BRIDGE BRIDGE_SHA256
#   WORLDS TURNS ARMS (';'-separated labor_posture_probe --arm specs; empty = defaults)
#   SAMPLING_SEED PLAYERS WIDTH HEIGHT SEPARATION JOBS THREADS JOB_TIMEOUT UPLOAD_RESERVE_S
#   SAMPLE_S MIN_MEM_GB
#
# Launch (detached):
#   SHA=<commit> setsid nohup bash $HOME/posture_probe_job.sh > $HOME/r5/posture-probe.out 2>&1 < /dev/null &
set -euo pipefail

JOB_TIMEOUT="${JOB_TIMEOUT:-120m}"
if [ "${POSTURE_UNDER_TIMEOUT:-0}" != "1" ]; then
  export POSTURE_UNDER_TIMEOUT=1
  exec timeout --signal=TERM --kill-after=120 "$JOB_TIMEOUT" bash "$0" "$@"
fi

SHA="${SHA:-__FILL_PROBE_SHA__}"
case "$SHA" in *__*) echo "unfilled SHA (export SHA=<probe code commit>)" >&2; exit 2 ;; esac
LAUNCH=$(date -u +%Y%m%dT%H%M%SZ)
GCS_ROOT="${GCS_ROOT:-gs://anthropocene-rl-datasets-508912/model9/r5}"
CODE_GCS="${CODE_GCS:-$GCS_ROOT/code/rl-pilot-${SHA:0:7}.tar.gz}"
CKPT_GCS="${CKPT_GCS:-$GCS_ROOT/d7-run/checkpoint-start.pt}"
CKPT_SHA256="${CKPT_SHA256:-14e749b0a50415f9f1b95d306bf3d8c19a810a29ef0f2755181d856566ba6483}"
DEST="${DEST:-$GCS_ROOT/ppo-pilot/posture-probe/$LAUNCH}"
WORK="${WORK:-$HOME/r5/posture-probe}"
CODEROOT="${CODEROOT:-$HOME/code/rl-pilot-${SHA:0:7}}"
CODE="$CODEROOT/rl"
PY="${PY:-$HOME/m9-bench/venv/bin/python}"
BRIDGE="${BRIDGE:-$HOME/code/r5s841c38e/rl/integration/bin/model9/Anthropocene.Model9Bridge.dll}"
BRIDGE_SHA256="${BRIDGE_SHA256:-99131dce9c7f0d0d66988b05c3c3e9adea6f6ad3407ce4cd0366d6a14b25eb6b}"
WORLDS="${WORLDS:-91300,91301}"
TURNS="${TURNS:-16}"
ARMS="${ARMS:-}"
SAMPLING_SEED="${SAMPLING_SEED:-0}"
PLAYERS="${PLAYERS:-16}"; WIDTH="${WIDTH:-128}"; HEIGHT="${HEIGHT:-64}"; SEPARATION="${SEPARATION:-9}"
JOBS="${JOBS:-7}"
THREADS="${THREADS:-1}"
UPLOAD_RESERVE_S="${UPLOAD_RESERVE_S:-300}"
SAMPLE_S="${SAMPLE_S:-15}"
MIN_MEM_GB="${MIN_MEM_GB:-16}"
START_S=$(date +%s)

ts() { date -u +%FT%TZ; }
mkdir -p "$WORK/logs" "$WORK/out" "$WORK/ckpt"
LOG_FILE="$WORK/logs/job-$LAUNCH.log"
OUT="$WORK/out/posture-probe-$LAUNCH.json"
CUR_STAGE=startup
log() { echo "$(ts) [posture-probe] $*" | tee -a "$LOG_FILE" >&2; }
die() { log "FAIL[$CUR_STAGE] $*"; exit 1; }
sha256_of() { sha256sum "$1" | cut -d' ' -f1; }
gcs_put() { timeout 600 gcloud storage cp --no-user-output-enabled "$1" "$2" < /dev/null || { log "upload failed: $1 -> $2"; return 1; }; }
remote_has() { timeout 120 gcloud storage ls "$1" > /dev/null 2>&1 < /dev/null; }
mem_available_gb() { awk '/MemAvailable/{a=$2} /MemFree/{f=$2} END{print int((a ? a : f) / 1048576)}' /proc/meminfo; }
seconds_of() {  # 60m / 2h / 90s / 120 -> seconds
  case "$1" in *h) echo $(( ${1%h} * 3600 )) ;; *m) echo $(( ${1%m} * 60 )) ;; *s) echo "${1%s}" ;; *) echo "$1" ;; esac
}

exec 9> "$WORK/job.lock"
flock -n 9 || { log "another posture_probe_job.sh holds $WORK/job.lock; exiting"; exit 3; }
if pgrep -f "[l]abor_posture_probe .*$WORK/out/" > /dev/null; then
  log "a posture probe is already running: $(pgrep -af "[l]abor_posture_probe .*$WORK/out/" | head -2); exiting"
  exit 3
fi
if remote_has "$DEST/EXIT_STATUS"; then
  log "$DEST/EXIT_STATUS already exists; nothing to do (use a new DEST to rerun)"
  exit 0
fi

SKIP_STATUS=0
SAMPLER_PID=""
finish() {
  local status=$?
  trap - EXIT
  set +e
  [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2> /dev/null
  kill_observed
  if [ "$SKIP_STATUS" != "1" ] && [ "$status" -ne 0 ] && [ ! -f "$WORK/EXIT_STATUS-$LAUNCH" ]; then
    echo "FAIL:$CUR_STAGE $(ts)" > "$WORK/EXIT_STATUS-$LAUNCH"
  fi
  gcs_put "$LOG_FILE" "$DEST/logs/job-$LAUNCH.log"
  if [ "$SKIP_STATUS" != "1" ] && [ -f "$WORK/EXIT_STATUS-$LAUNCH" ]; then
    if remote_has "$DEST/EXIT_STATUS"; then
      log "$DEST/EXIT_STATUS appeared during this launch; not overwriting"
    else
      gcs_put "$WORK/EXIT_STATUS-$LAUNCH" "$DEST/EXIT_STATUS"   # last and once; never stops the VM
    fi
  fi
  exit "$status"
}
trap finish EXIT
trap 'SKIP_STATUS=1; log "signal received: interrupted, no EXIT_STATUS (rerun the same line)"; exit 143' TERM INT HUP

# --- process-tree accounting (as anchor_probe_job.sh) --------------------------------------
tree_pids() {
  local all="$1" frontier="$1" next
  while [ -n "$frontier" ]; do
    next=$(pgrep -P "$(echo $frontier | tr ' ' ',')" 2> /dev/null | tr '\n' ' ' || true)
    next="${next% }"
    [ -n "$next" ] || break
    all="$all $next"; frontier="$next"
  done
  echo "$all"
}
sample_loop() {  # ROOT_PID ; appends epoch,processes,rss_kb,threads,pcpu,load1 to samples-<launch>.csv
  local root=$1 pids line
  while kill -0 "$root" 2> /dev/null; do
    pids=$(tree_pids "$root")
    ps -o pid=,lstart= -p "$(echo $pids | tr ' ' ',')" 2> /dev/null >> "$WORK/observed-pids-$LAUNCH.txt" || true
    line=$(ps -o rss=,nlwp=,pcpu= -p "$(echo $pids | tr ' ' ',')" 2> /dev/null \
      | awk '{n++; r+=$1; t+=$2; c+=$3} END {printf "%d,%d,%d,%.1f", n, r, t, c}')
    echo "$(date +%s),$line,$(cut -d' ' -f1 /proc/loadavg),$(mem_available_gb)" >> "$WORK/samples-$LAUNCH.csv"
    sleep "$SAMPLE_S"
  done
}
kill_observed() {  # only still-alive processes seen in our probe's tree (same pid AND start time)
  local f="$WORK/observed-pids-$LAUNCH.txt" entry pid now comm
  [ -f "$f" ] || return 0
  while IFS= read -r entry; do
    pid=$(echo "$entry" | awk '{print $1}')
    [ -n "$pid" ] || continue
    now=$(ps -o pid=,lstart= -p "$pid" 2> /dev/null || true)
    [ -n "$now" ] && [ "$now" = "$entry" ] || continue
    comm=$(ps -o comm= -p "$pid" 2> /dev/null || true)
    case "$comm" in python*|dotnet*|Anthropocene*|timeout|nice) kill -KILL "$pid" 2> /dev/null && log "killed leftover $comm $pid" ;; esac
  done < <(sort -u "$f")
}

tar_commit_id() {
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
    mv "$CODEROOT.tmp" "$CODEROOT"
    touch "$CODEROOT/.snapshot-ok"
  fi
  SOURCE_COMMIT=$(cat "$CODEROOT/COMMIT")
  export PYTHONPATH="$CODE/src" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  export ANTHROPOCENE_DOTNET="${ANTHROPOCENE_DOTNET:-/usr/bin/dotnet}"
  export ANTHROPOCENE_MODEL9_BRIDGE="$BRIDGE"
  [ -f "$BRIDGE" ] || die "missing bridge $BRIDGE"
  [ "$(sha256_of "$BRIDGE")" = "$BRIDGE_SHA256" ] || die "bridge sha256 differs (set BRIDGE_SHA256 if a rebuild is intended)"
  [ -x "$ANTHROPOCENE_DOTNET" ] || die "missing dotnet $ANTHROPOCENE_DOTNET"
  local where help_text
  where=$(cd "$CODE" && "$PY" -c 'import anthropocene_rl,sys; print(anthropocene_rl.__file__)' 2>&1) || die "cannot import anthropocene_rl: $where"
  case "$where" in "$CODE"/src/*) ;; *) die "anthropocene_rl resolves to $where, not $CODE/src" ;; esac
  help_text=$(cd "$CODE" && "$PY" -m anthropocene_rl.autonomous_v7.labor_posture_probe --help < /dev/null 2>&1 || true)
  case "$help_text" in *--arm*--jobs*--threads*--progress*) ;; *) die "code $SOURCE_COMMIT lacks labor_posture_probe --arm/--jobs/--threads/--progress" ;; esac
  log "code $SOURCE_COMMIT; bridge ${BRIDGE_SHA256:0:16}; free mem $(mem_available_gb) GB; cpus $(getconf _NPROCESSORS_ONLN); load $(cut -d' ' -f1-3 /proc/loadavg)"
}

stage_checkpoint() {
  CUR_STAGE=checkpoint
  CKPT="$WORK/ckpt/checkpoint-${CKPT_SHA256:0:16}.pt"
  if [ ! -f "$CKPT" ] || [ "$(sha256_of "$CKPT")" != "$CKPT_SHA256" ]; then
    rm -f "$CKPT"
    timeout 900 gcloud storage cp --no-user-output-enabled "$CKPT_GCS" "$CKPT" < /dev/null || die "cannot fetch $CKPT_GCS"
  fi
  [ "$(sha256_of "$CKPT")" = "$CKPT_SHA256" ] || die "checkpoint sha256 $(sha256_of "$CKPT") != $CKPT_SHA256"
  (cd "$CODE" && "$PY" - "$CKPT" "$WORLDS" <<'PY') >> "$LOG_FILE" 2>&1 || die "checkpoint/world preflight failed (see job log)"
import sys, torch
from anthropocene_rl.autonomous_v7 import labor_training as T
from anthropocene_rl.autonomous_v7.labor_worlds import require_training_worlds
from anthropocene_rl.autonomous_v7.protocol import seeds as checked
state = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
T._require_trainer_checkpoint(state)
T._models_from_checkpoint(state)
worlds = [int(w) for w in sys.argv[2].split(",") if w]
assert worlds and len(set(worlds)) == len(worlds), "probe worlds must be distinct"
require_training_worlds(worlds, "posture probe")
checked(worlds, "posture probe worlds")  # sealed 3000-3059 refused
used = set(range(91000, 91006)) | {91100 + 10 * k + i for k in range(10) for i in range(10)} \
    | {91200 + 10 * k + i for k in range(10) for i in range(10)}
assert not set(worlds) & used, "probe world collides with D7/step-check or a pilot block"
tempered = state.get("headTempering") or []
print("preflight ok: iteration", state["iteration"], "worlds", worlds,
      "tempered", [r.get("transform") for r in tempered])
PY
  log "checkpoint ${CKPT_SHA256:0:16} ok; $(tail -1 "$LOG_FILE")"
}

stage_run() {
  CUR_STAGE=run
  local budget=$(( $(seconds_of "$JOB_TIMEOUT") - ($(date +%s) - START_S) - UPLOAD_RESERVE_S ))
  [ "$budget" -gt 120 ] || die "no time budget left ($budget s)"
  [ "$(mem_available_gb)" -ge "$MIN_MEM_GB" ] || die "MemAvailable $(mem_available_gb) GB < $MIN_MEM_GB GB (the pilot shares this VM)"
  local arms=() spec
  if [ -n "$ARMS" ]; then
    IFS=';' read -ra specs <<< "$ARMS"
    for spec in "${specs[@]}"; do [ -n "$spec" ] && arms+=(--arm "$spec"); done
  fi
  log "probe: worlds $WORLDS turns $TURNS arms ${ARMS:-default} jobs $JOBS threads $THREADS; budget ${budget}s; mem $(mem_available_gb) GB; load $(cut -d' ' -f1 /proc/loadavg)"
  RUN_START=$(date +%s)
  (cd "$CODE" && exec nice -n 10 timeout --signal=TERM --kill-after=30 "$budget" \
      "$PY" -m anthropocene_rl.autonomous_v7.labor_posture_probe \
      --checkpoint "$CKPT" --worlds "$WORLDS" --turns "$TURNS" "${arms[@]}" \
      --sampling-seed "$SAMPLING_SEED" --players "$PLAYERS" --width "$WIDTH" --height "$HEIGHT" \
      --separation "$SEPARATION" --jobs "$JOBS" --threads "$THREADS" --progress \
      --out "$OUT" > "$WORK/logs/probe-$LAUNCH.log" 2>&1 < /dev/null) &
  local probe_pid=$!
  sample_loop "$probe_pid" &
  SAMPLER_PID=$!
  PROBE_RC=0
  wait "$probe_pid" || PROBE_RC=$?
  PROBE_SECONDS=$(( $(date +%s) - RUN_START ))
  kill "$SAMPLER_PID" 2> /dev/null || true; SAMPLER_PID=""
  sleep 5; kill_observed
  log "probe rc=$PROBE_RC after ${PROBE_SECONDS}s (124/137 = hit the ${budget}s cap)"
  if [ "$PROBE_RC" -ne 0 ]; then
    CUR_STAGE=probe
    local unit n=0
    for unit in "$OUT.units"/*.json; do  # every finished (arm, world) unit, scope-tagged
      [ -f "$unit" ] || continue
      gcs_put "$unit" "$DEST/partial/$(basename "$unit")" && n=$((n + 1))
    done
    gcs_put "$WORK/logs/probe-$LAUNCH.log" "$DEST/logs/probe-$LAUNCH.log" || true
    die "the probe failed or hit its cap (rc $PROBE_RC); $n finished units uploaded to $DEST/partial/"
  fi
  [ -f "$OUT" ] || { CUR_STAGE=probe; die "the probe wrote no output"; }
  "$PY" - "$OUT" <<'PY' >> "$LOG_FILE" 2>&1 || { CUR_STAGE=probe-check; die "probe output check failed"; }
import json, sys
d = json.load(open(sys.argv[1]))
assert d["scope"] == "diagnostic-posture-override", d.get("scope")
for arm, s in d["summary"].items():
    print(arm, {k: s[k] for k in ("decisionTurns", "campDecisions", "campsWithTrade", "tradeTotals",
                                  "fissionDaughters", "terminated", "vanished", "finalCamps",
                                  "foundersAlive", "meanQolReward", "meanPopulationReward")})
    f = s["feasibility"]
    print("  feasibility", {k: f[k] for k in ("rejectedFirstProposals", "rejectedShare",
                                            "retriesStillInvalid", "fallbacks")},
          "initial", f["feasibilityInitial"], "guard", f["settlementGuard"],
          "reasons", f["initialViolationReasons"])
PY
}

stage_upload() {
  CUR_STAGE=upload
  local f
  for f in "$WORK/logs/probe-$LAUNCH.log" "$WORK/samples-$LAUNCH.csv"; do
    [ -f "$f" ] && gcs_put "$f" "$DEST/logs/$(basename "$f")"
  done
  sha256_of "$OUT" > "$OUT.sha256"
  gcs_put "$OUT" "$DEST/$(basename "$OUT")" || die "probe output upload failed"
  gcs_put "$OUT.sha256" "$DEST/$(basename "$OUT").sha256" || die "probe sha upload failed"
  echo "0 $(ts) posture-probe source=$SOURCE_COMMIT ckpt=${CKPT_SHA256:0:16} worlds=$WORLDS turns=$TURNS jobs=$JOBS seconds=$PROBE_SECONDS out=$(basename "$OUT")" > "$WORK/EXIT_STATUS-$LAUNCH"
  log "done; EXIT_STATUS 0 (uploaded last by the exit handler); the VM is left running"
}

log "start launch=$LAUNCH sha=$SHA ckpt=${CKPT_SHA256:0:16} worlds=$WORLDS turns=$TURNS reset=$PLAYERS/${WIDTH}x$HEIGHT/sep$SEPARATION/movement jobs=$JOBS threads=$THREADS dest=$DEST cap=$JOB_TIMEOUT"
rm -f "$WORK/observed-pids-$LAUNCH.txt"
stage_code
stage_checkpoint
stage_run
stage_upload
