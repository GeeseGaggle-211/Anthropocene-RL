#!/usr/bin/env bash
# r5 PPO pilot: anchor timing probe (CPU VM anthropocene-trainer-60, c3d-highcpu-60 Spot,
# us-central1-c). Measures what one labor_anchor_producer world costs at DEFAULT anchor
# settings (prefix 25, 16 geometric tail draws, rho 0.95, cap 200, 4 paths x 2 continuations)
# on the training worlds (16 players, 128x64, separation 9, movement), from the D7 start
# checkpoint (sha 14e749b0...), so the pilot can be costed. It trains nothing.
#
# Two producers run CONCURRENTLY (independent; same checkpoint):
#   train-anchor on world TRAIN_WORLD (default 91190), actor-gate on GATE_WORLD (91290).
# Fresh worlds: not in the critic-training history (90100-90165), not D7/step-check
# (91000-91005), not the pilot's rollout/gate blocks (91100-91182, 91200-91281), never
# sealed 3000-3059 or reserved 90154-90159 (checked in the preflight).
#
# Parallelism (what the producer allows): labor_anchor_producer --jobs N (added for the
# pilot) runs each (world, path) unit in its own spawned process with its own native bridge:
# a fresh reset of the world, a fork, then that path's two continuations (sampled-policy and
# paired-sample). One world has --paths units (default 4), so JOBS=4 per producer is the
# maximum useful parallelism here (8 processes + their dotnet sims for both producers). The
# payload is the sequential one (same seeds/schedules/rows); every unit's reset decision must
# be identical or the producer refuses (a reproducibility check of native reset across
# processes, itself a probe result). --threads pins torch threads per process (THREADS).
# Without --jobs (JOBS=1) a world's 8 continuations run in sequence on one bridge.
#
# Hard cap: the whole job re-execs under `timeout` (JOB_TIMEOUT, default 40m); the producers
# get the budget minus UPLOAD_RESERVE_S and are expected to hit it at default settings (a
# world is ~550-850 native steps). That is not a failure: --progress writes one stderr line
# per PROGRESS_EVERY_TERMS(5) native steps of every continuation plus the frozen work plan, so
# the summary extrapolates the full-world wall time from measured seconds per step. A
# producer that finishes uploads its .npz too. Peak RSS / threads / processes of each
# producer's whole process tree (python main, spawned workers, dotnet sims) are sampled every
# SAMPLE_S seconds; leftover processes of the tree are killed after a cap.
#
# Conventions follow ops/r5/step-check/collect_job.sh: code tarball (git archive, commit-id
# check), shared venv, shared bridge (sha256 99131dce...), OMP/MKL 1 thread, nice 10, flock +
# pgrep duplicate guards, launch-stamped logs, EXIT_STATUS uploaded once and LAST
# ("0 <utc> ..." or "FAIL:<stage> <utc>"). It NEVER stops the VM, touches no crontab and no
# other job's WORK/DEST, and kills only processes it observed in its own producers' trees.
#
# Required env: SHA (the pilot code commit; the tarball must be uploaded first).
# Optional: GCS_ROOT CODE_GCS CKPT_GCS CKPT_SHA256 DEST WORK CODEROOT PY BRIDGE BRIDGE_SHA256
#   TRAIN_WORLD GATE_WORLD TRAIN_SEED_BASE GATE_SEED_BASE OPTIMIZER_SEED PLAYERS WIDTH HEIGHT
#   SEPARATION JOBS THREADS JOB_TIMEOUT UPLOAD_RESERVE_S SAMPLE_S MIN_MEM_GB
#
# Launch (detached; a rerun after a VM stop starts over unless EXIT_STATUS exists):
#   setsid nohup bash $HOME/anchor_probe_job.sh > $HOME/r5/anchor-probe.out 2>&1 < /dev/null &
set -euo pipefail

JOB_TIMEOUT="${JOB_TIMEOUT:-40m}"
if [ "${PROBE_UNDER_TIMEOUT:-0}" != "1" ]; then
  export PROBE_UNDER_TIMEOUT=1
  exec timeout --signal=TERM --kill-after=120 "$JOB_TIMEOUT" bash "$0" "$@"
fi

SHA="${SHA:-__FILL_PILOT_SHA__}"
GCS_ROOT="${GCS_ROOT:-gs://anthropocene-rl-datasets-508912/model9/r5}"
CODE_GCS="${CODE_GCS:-$GCS_ROOT/code/rl-pilot-${SHA:0:7}.tar.gz}"
CKPT_GCS="${CKPT_GCS:-$GCS_ROOT/d7-run/checkpoint-start.pt}"
CKPT_SHA256="${CKPT_SHA256:-14e749b0a50415f9f1b95d306bf3d8c19a810a29ef0f2755181d856566ba6483}"
DEST="${DEST:-$GCS_ROOT/ppo-pilot/anchor-probe}"
WORK="${WORK:-$HOME/r5/ppo-pilot-probe}"
CODEROOT="${CODEROOT:-$HOME/code/rl-pilot-${SHA:0:7}}"
CODE="$CODEROOT/rl"
PY="${PY:-$HOME/m9-bench/venv/bin/python}"
BRIDGE="${BRIDGE:-$HOME/code/r5s841c38e/rl/integration/bin/model9/Anthropocene.Model9Bridge.dll}"
BRIDGE_SHA256="${BRIDGE_SHA256:-99131dce9c7f0d0d66988b05c3c3e9adea6f6ad3407ce4cd0366d6a14b25eb6b}"
TRAIN_WORLD="${TRAIN_WORLD:-91190}"
GATE_WORLD="${GATE_WORLD:-91290}"
TRAIN_SEED_BASE="${TRAIN_SEED_BASE:-9190000}"
GATE_SEED_BASE="${GATE_SEED_BASE:-9290000}"
OPTIMIZER_SEED="${OPTIMIZER_SEED:-0}"
PLAYERS="${PLAYERS:-16}"; WIDTH="${WIDTH:-128}"; HEIGHT="${HEIGHT:-64}"; SEPARATION="${SEPARATION:-9}"
JOBS="${JOBS:-4}"
THREADS="${THREADS:-1}"
UPLOAD_RESERVE_S="${UPLOAD_RESERVE_S:-300}"
SAMPLE_S="${SAMPLE_S:-15}"
MIN_MEM_GB="${MIN_MEM_GB:-16}"
case "$SHA" in *__*) echo "unfilled SHA (export SHA=<pilot code commit>)" >&2; exit 2 ;; esac
START_S=$(date +%s)

ts() { date -u +%FT%TZ; }
LAUNCH=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$WORK/logs" "$WORK/out" "$WORK/ckpt"
LOG_FILE="$WORK/logs/job-$LAUNCH.log"
CUR_STAGE=startup
log() { echo "$(ts) [anchor-probe] $*" | tee -a "$LOG_FILE" >&2; }
die() { log "FAIL[$CUR_STAGE] $*"; exit 1; }
sha256_of() { sha256sum "$1" | cut -d' ' -f1; }
gcs_put() { timeout 600 gcloud storage cp --no-user-output-enabled "$1" "$2" < /dev/null || { log "upload failed: $1 -> $2"; return 1; }; }
remote_has() { timeout 120 gcloud storage ls "$1" > /dev/null 2>&1 < /dev/null; }
mem_available_gb() { awk '/MemAvailable/{a=$2} /MemFree/{f=$2} END{print int((a ? a : f) / 1048576)}' /proc/meminfo; }
seconds_of() {  # 40m / 2h / 90s / 120 -> seconds
  case "$1" in *h) echo $(( ${1%h} * 3600 )) ;; *m) echo $(( ${1%m} * 60 )) ;; *s) echo "${1%s}" ;; *) echo "$1" ;; esac
}

exec 9> "$WORK/job.lock"
flock -n 9 || { log "another anchor_probe_job.sh holds $WORK/job.lock; exiting"; exit 3; }
if pgrep -f "[l]abor_anchor_producer .*$WORK/out/" > /dev/null; then
  log "a probe producer is already running: $(pgrep -af "[l]abor_anchor_producer .*$WORK/out/" | head -2); exiting"
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
  if [ "$SKIP_STATUS" != "1" ] && [ "$status" -ne 0 ] && [ ! -f "$WORK/EXIT_STATUS" ]; then
    echo "FAIL:$CUR_STAGE $(ts)" > "$WORK/EXIT_STATUS"
  fi
  gcs_put "$LOG_FILE" "$DEST/logs/job-$LAUNCH.log"
  if [ "$SKIP_STATUS" != "1" ] && [ -f "$WORK/EXIT_STATUS" ]; then
    if remote_has "$DEST/EXIT_STATUS"; then
      log "$DEST/EXIT_STATUS appeared during this launch; not overwriting (local: $(cat "$WORK/EXIT_STATUS"))"
    else
      gcs_put "$WORK/EXIT_STATUS" "$DEST/EXIT_STATUS"   # last and once; this job never stops the VM
    fi
  fi
  exit "$status"
}
trap finish EXIT
trap 'SKIP_STATUS=1; log "signal received: interrupted, no EXIT_STATUS (rerun the same line)"; exit 143' TERM INT HUP

# --- process-tree accounting ------------------------------------------------------------
tree_pids() {  # ROOT -> ROOT and every descendant pid (space separated)
  local all="$1" frontier="$1" next
  while [ -n "$frontier" ]; do
    next=$(pgrep -P "$(echo $frontier | tr ' ' ',')" 2> /dev/null | tr '\n' ' ' || true)
    next="${next% }"
    [ -n "$next" ] || break
    all="$all $next"; frontier="$next"
  done
  echo "$all"
}
sample_loop() {  # ROLE:PID ... ; appends epoch,role,processes,rss_kb,threads,pcpu to samples.csv
  local pair role root pids line
  while true; do
    for pair in "$@"; do
      role="${pair%%:*}"; root="${pair##*:}"
      kill -0 "$root" 2> /dev/null || continue
      pids=$(tree_pids "$root")
      # pid plus its start time: a later kill never hits a recycled pid
      ps -o pid=,lstart= -p "$(echo $pids | tr ' ' ',')" 2> /dev/null >> "$WORK/observed-pids-$role.txt" || true
      line=$(ps -o rss=,nlwp=,pcpu= -p "$(echo $pids | tr ' ' ',')" 2> /dev/null \
        | awk '{n++; r+=$1; t+=$2; c+=$3} END {printf "%d,%d,%d,%.1f", n, r, t, c}')
      echo "$(date +%s),$role,$line,$(cut -d' ' -f1 /proc/loadavg)" >> "$WORK/samples-$LAUNCH.csv"
    done
    sleep "$SAMPLE_S"
  done
}
kill_observed() {  # kill only still-alive processes seen in our producers' trees (same pid AND start time)
  local f entry pid now comm
  for f in "$WORK"/observed-pids-*.txt; do
    [ -f "$f" ] || continue
    while IFS= read -r entry; do
      pid=$(echo "$entry" | awk '{print $1}')
      [ -n "$pid" ] || continue
      now=$(ps -o pid=,lstart= -p "$pid" 2> /dev/null || true)
      [ -n "$now" ] && [ "$now" = "$entry" ] || continue
      comm=$(ps -o comm= -p "$pid" 2> /dev/null || true)
      case "$comm" in python*|dotnet*|Anthropocene*|timeout|time|nice) kill -KILL "$pid" 2> /dev/null && log "killed leftover $comm $pid" ;; esac
    done < <(sort -u "$f")
  done
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
  help_text=$(cd "$CODE" && "$PY" -m anthropocene_rl.autonomous_v7.labor_anchor_producer --help < /dev/null 2>&1 || true)
  case "$help_text" in *--jobs*--threads*--progress*) ;; *) die "code $SOURCE_COMMIT lacks labor_anchor_producer --jobs/--threads/--progress" ;; esac
  log "code $SOURCE_COMMIT; bridge ${BRIDGE_SHA256:0:16}; free mem $(mem_available_gb) GB; cpus $(getconf _NPROCESSORS_ONLN); load $(cut -d' ' -f1-3 /proc/loadavg)"
}

stage_checkpoint() {
  CUR_STAGE=checkpoint
  CKPT="$WORK/ckpt/checkpoint-start.pt"
  if [ ! -f "$CKPT" ] || [ "$(sha256_of "$CKPT")" != "$CKPT_SHA256" ]; then
    rm -f "$CKPT"
    timeout 900 gcloud storage cp --no-user-output-enabled "$CKPT_GCS" "$CKPT" < /dev/null || die "cannot fetch $CKPT_GCS"
  fi
  [ "$(sha256_of "$CKPT")" = "$CKPT_SHA256" ] || die "checkpoint sha256 $(sha256_of "$CKPT") != $CKPT_SHA256"
  (cd "$CODE" && "$PY" - "$CKPT" "$TRAIN_WORLD" "$GATE_WORLD" "$TRAIN_SEED_BASE" "$GATE_SEED_BASE" <<'PY') >> "$LOG_FILE" 2>&1 || die "checkpoint/world preflight failed (see job log)"
import sys, torch
from anthropocene_rl.autonomous_v7 import labor_training as T
from anthropocene_rl.autonomous_v7.labor_anchor_producer import _path_rng_seeds
from anthropocene_rl.autonomous_v7.labor_worlds import require_training_worlds
from anthropocene_rl.autonomous_v7.protocol import seeds as checked
state = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
T._require_trainer_checkpoint(state)
T._models_from_checkpoint(state)
train, gate, train_base, gate_base = (int(v) for v in sys.argv[2:6])
require_training_worlds([train, gate], "anchor probe")
checked([train, gate], "anchor probe worlds")  # sealed 3000-3059 refused
history = set(state["critic_training_world_seeds"])
assert train not in history and gate not in history, "probe world in the critic-training history"
assert train != gate
used = set(range(91000, 91006)) | {91100 + 10 * k + i for k in range(9) for i in range(10)} \
    | {91200 + 10 * k + i for k in range(9) for i in range(10)}
assert not {train, gate} & used, "probe world collides with D7/step-check or a pilot block"
for base in (train_base, gate_base):
    checked([s for p in range(4) for s in _path_rng_seeds(base, 0, p)], "anchor probe seeds")
print("preflight ok: iteration", state["iteration"], "history", len(history), "worlds", train, gate)
PY
  log "checkpoint ${CKPT_SHA256:0:16} ok; $(tail -1 "$LOG_FILE")"
}

stage_run() {
  CUR_STAGE=run
  local budget=$(( $(seconds_of "$JOB_TIMEOUT") - ($(date +%s) - START_S) - UPLOAD_RESERVE_S ))
  [ "$budget" -gt 120 ] || die "no time budget left ($budget s)"
  [ "$(mem_available_gb)" -ge "$MIN_MEM_GB" ] || die "MemAvailable $(mem_available_gb) GB < $MIN_MEM_GB GB"
  local common=(--checkpoint "$CKPT" --optimizer-seed "$OPTIMIZER_SEED" --allow-vm-scale
    --players "$PLAYERS" --width "$WIDTH" --height "$HEIGHT" --separation "$SEPARATION"
    --jobs "$JOBS" --threads "$THREADS" --progress)
  log "producers: train-anchor world $TRAIN_WORLD, actor-gate world $GATE_WORLD; jobs $JOBS threads $THREADS; budget ${budget}s; mem $(mem_available_gb) GB"
  local time_train=() time_gate=()
  if [ -x /usr/bin/time ]; then  # GNU time: the main process's max RSS (the tree is sampled)
    time_train=(/usr/bin/time -v -o "$WORK/logs/time-train-$LAUNCH.txt")
    time_gate=(/usr/bin/time -v -o "$WORK/logs/time-gate-$LAUNCH.txt")
  fi
  RUN_START=$(date +%s)
  (cd "$CODE" && exec nice -n 10 "${time_train[@]}" \
      timeout --signal=TERM --kill-after=30 "$budget" "$PY" -m anthropocene_rl.autonomous_v7.labor_anchor_producer \
      "${common[@]}" --worlds "$TRAIN_WORLD" --continuation-seed-base "$TRAIN_SEED_BASE" --role train-anchor \
      --out "$WORK/out/train-anchor-$TRAIN_WORLD.npz" > "$WORK/logs/train-anchor-$LAUNCH.log" 2>&1 < /dev/null) &
  local train_pid=$!
  (cd "$CODE" && exec nice -n 10 "${time_gate[@]}" \
      timeout --signal=TERM --kill-after=30 "$budget" "$PY" -m anthropocene_rl.autonomous_v7.labor_anchor_producer \
      "${common[@]}" --worlds "$GATE_WORLD" --continuation-seed-base "$GATE_SEED_BASE" --role actor-gate \
      --out "$WORK/out/actor-gate-$GATE_WORLD.npz" > "$WORK/logs/actor-gate-$LAUNCH.log" 2>&1 < /dev/null) &
  local gate_pid=$!
  sample_loop "train:$train_pid" "gate:$gate_pid" &
  SAMPLER_PID=$!
  TRAIN_RC=0; GATE_RC=0
  wait "$train_pid" || TRAIN_RC=$?
  TRAIN_SECONDS=$(( $(date +%s) - RUN_START ))
  wait "$gate_pid" || GATE_RC=$?
  GATE_SECONDS=$(( $(date +%s) - RUN_START ))
  kill "$SAMPLER_PID" 2> /dev/null || true; SAMPLER_PID=""
  sleep 5; kill_observed
  log "train-anchor rc=$TRAIN_RC after ${TRAIN_SECONDS}s; actor-gate rc=$GATE_RC after ${GATE_SECONDS}s (124/137 = hit the ${budget}s cap)"
  for rc in "$TRAIN_RC" "$GATE_RC"; do
    case "$rc" in 0|124|137) ;; *) CUR_STAGE=producer; die "a producer failed (rc $rc); see logs" ;; esac
  done
}

stage_summary() {
  CUR_STAGE=summary
  (cd "$CODE" && "$PY" - "$WORK" "$LAUNCH" "$TRAIN_RC" "$GATE_RC" "$TRAIN_SECONDS" "$GATE_SECONDS" \
      "$JOBS" "$THREADS" "$TRAIN_WORLD" "$GATE_WORLD" <<'PY') > "$WORK/out/probe-summary-$LAUNCH.json" 2>> "$LOG_FILE" || die "summary failed"
import csv, json, re, sys
from pathlib import Path
work, launch = Path(sys.argv[1]), sys.argv[2]
rcs = {"train": int(sys.argv[3]), "gate": int(sys.argv[4])}
wall = {"train": int(sys.argv[5]), "gate": int(sys.argv[6])}
jobs, threads = int(sys.argv[7]), int(sys.argv[8])
worlds = {"train": int(sys.argv[9]), "gate": int(sys.argv[10])}
logs = {"train": work / f"logs/train-anchor-{launch}.log", "gate": work / f"logs/actor-gate-{launch}.log"}
outs = {"train": work / f"out/train-anchor-{worlds['train']}.npz", "gate": work / f"out/actor-gate-{worlds['gate']}.npz"}
samples = list(csv.reader(open(work / f"samples-{launch}.csv"))) if (work / f"samples-{launch}.csv").exists() else []

def lines(path, key):
    out = []
    for raw in path.read_text(errors="replace").splitlines() if path.exists() else []:
        if raw.startswith('{"' + key + '"'):
            out.append(json.loads(raw)[key])
    return out

def peak_rss(path):
    if not path.exists():
        return None
    match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", path.read_text())
    return None if match is None else int(match.group(1)) / 1024

result = {"schema": "r5-ppo-pilot-anchor-probe-v1", "jobs": jobs, "threads": threads, "roles": {}}
for role in ("train", "gate"):
    plan = lines(logs[role], "anchorPlan")
    done = lines(logs[role], "anchorProgress")
    terms = lines(logs[role], "anchorTerm")
    # latest term line per continuation (finished continuations: their anchorProgress line)
    latest = {}
    for item in terms:
        latest[(item["world"], item["path"], item["alternative"])] = item
    for item in done:
        latest[(item["world"], item["path"], item["alternative"])] = item
    measured_terms = sum(item["terms"] for item in latest.values())
    measured_seconds = sum(item["seconds"] for item in latest.values())
    per_term = measured_seconds / measured_terms if measured_terms else None
    maximum = plan[0]["maximumTerms"] if plan else {}
    # upper bound: no camp terminates before its frozen schedule's last required term
    sequential = None if per_term is None else per_term * 2 * sum(maximum.values())
    parallel = None if per_term is None else per_term * 2 * max(maximum.values(), default=0)
    rows = [r for r in samples if r[1] == role]
    result["roles"][role] = {
        "world": worlds[role],
        "rc": rcs[role],
        "completed": rcs[role] == 0 and outs[role].exists(),
        "hitCap": rcs[role] in (124, 137),
        "wallSeconds": wall[role],
        "plan": plan[0] if plan else None,
        "continuationsFinished": len(done),
        "continuationsStarted": len(latest),
        "measuredNativeSteps": measured_terms,
        "secondsPerNativeStep": per_term,
        "perContinuation": sorted(latest.values(), key=lambda x: (x["path"], x["alternative"])),
        "extrapolatedWorldSecondsUpperBound": {
            "sequentialJobs1": sequential,
            "parallelOneProcessPerPath": parallel,
            "assumption": "seconds/step measured here; every opening camp alive to its schedule's maximum term",
        },
        "peakTreeRssMB": max((int(r[3]) / 1024 for r in rows), default=None),
        "peakTreeThreads": max((int(r[4]) for r in rows), default=None),
        "peakTreeProcesses": max((int(r[2]) for r in rows), default=None),
        "peakTreeCpuPercent": max((float(r[5]) for r in rows), default=None),
        "mainProcessMaxRssMB": peak_rss(work / f"logs/time-{role}-{launch}.txt"),
    }
result["peakLoadAvg1"] = max((float(r[6]) for r in samples), default=None)
result["samples"] = len(samples)
print(json.dumps(result, indent=1, sort_keys=True))
PY
  log "summary: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print({k:(v["completed"],v["secondsPerNativeStep"],v["peakTreeRssMB"],v["peakTreeThreads"]) for k,v in d["roles"].items()})' "$WORK/out/probe-summary-$LAUNCH.json" 2>/dev/null || echo see file)"
}

stage_upload() {
  CUR_STAGE=upload
  local f
  for f in "$WORK/logs/train-anchor-$LAUNCH.log" "$WORK/logs/actor-gate-$LAUNCH.log" \
      "$WORK/logs/time-train-$LAUNCH.txt" "$WORK/logs/time-gate-$LAUNCH.txt" "$WORK/samples-$LAUNCH.csv"; do
    [ -f "$f" ] && gcs_put "$f" "$DEST/logs/$(basename "$f")"
  done
  for f in "$WORK/out/train-anchor-$TRAIN_WORLD.npz" "$WORK/out/actor-gate-$GATE_WORLD.npz"; do
    if [ -f "$f" ]; then
      sha256_of "$f" > "$f.sha256"
      gcs_put "$f" "$DEST/$(basename "$f")" && gcs_put "$f.sha256" "$DEST/$(basename "$f").sha256"
    fi
  done
  gcs_put "$WORK/out/probe-summary-$LAUNCH.json" "$DEST/probe-summary-$LAUNCH.json" || die "summary upload failed"
  echo "0 $(ts) probe source=$SOURCE_COMMIT train_rc=$TRAIN_RC gate_rc=$GATE_RC jobs=$JOBS threads=$THREADS summary=probe-summary-$LAUNCH.json" > "$WORK/EXIT_STATUS"
  log "done; EXIT_STATUS 0 (uploaded last by the exit handler); the VM is left running"
}

log "start launch=$LAUNCH sha=$SHA worlds train=$TRAIN_WORLD gate=$GATE_WORLD reset=$PLAYERS/${WIDTH}x$HEIGHT/sep$SEPARATION/movement jobs=$JOBS threads=$THREADS dest=$DEST cap=$JOB_TIMEOUT"
rm -f "$WORK/EXIT_STATUS" "$WORK"/observed-pids-*.txt
stage_code
stage_checkpoint
stage_run
stage_summary
stage_upload
