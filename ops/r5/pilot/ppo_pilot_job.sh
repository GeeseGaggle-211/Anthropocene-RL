#!/usr/bin/env bash
# r5 minimal PPO pilot (actor + critic, plain GAE, --a-mode off) on the CPU VM
# anthropocene-trainer-60 (c3d-highcpu-60 Spot, us-central1-c). A PILOT, not a readiness claim.
#
# One iteration k (runbook docs/MODEL9_LABOR_TRAINING_RUNBOOK.md sections 2-6), all from the
# iteration's input checkpoint C_k (C_0 = the D7 start checkpoint, sha 14e749b0...):
#   collect        labor_training collect, ROLLOUT_WORLDS worlds x TURNS decisions      } run
#   train-anchor   labor_anchor_producer --role train-anchor on the first                } concur-
#                  TRAIN_ANCHOR_WORLDS rollout worlds                                    } rently
#   actor-gate     labor_anchor_producer --role actor-gate on GATE_WORLDS held-out worlds} (all read
#                  (telemetry; GATE_TELEMETRY=1)                                         }  C_k only)
#   calibration    labor_value_calibration on the actor-gate anchor (GATE_TELEMETRY=1)
#   train          labor_training train --a-mode off with the adopted actor step settings
#                  (encoder lr 1e-7, head lr 3e-5, warm-up 12 Adam steps, target KL 0.02,
#                  4 epochs, batch 256), train's default critic lr 3e-4 / 1 critic epoch, and
#                  --pilot-ungated-actor (PILOT_BYPASS=1, default): the actor updates even
#                  when the four-head gate is closed (the expected outcome); the output
#                  checkpoint records actorGate "pilot-bypass" + the report's real status and
#                  the persistent field pilot_ungated_actor_updates, which acceptance tooling
#                  refuses. PILOT_BYPASS=0 is the gated runbook loop (critic-only when closed).
#   summary        labor_pilot summary -> summary.json (rollout parents, anchor rows, gate per
#                  head, critic losses, actor accepted/rolled-back steps, per-epoch KL, mean
#                  |dlogp| per head, wall seconds per stage)
# C_{k+1} = the train output. Anchors are bound to C_k's policy digest, so every iteration
# recollects both (they are stale after any actor step).
#
# Worlds and continuation seeds derive from k only (labor_pilot.pilot_plan): rollout and
# train-anchor 91100+10k.., actor-gate 91200+10k.., seed bases 10,000,000+100,000k (train
# anchor) and 20,000,000+100,000k (gate); k <= 8 (91190/91290 are the anchor probe's). The
# preflight refuses sealed (3000-3059) and reserved (90154-90159) worlds, gate worlds in the
# critic-training history or the rollout/anchor, rollout worlds already in the history, and a
# checkpoint whose iteration is not START_ITERATION + k. Every stage also re-checks its
# output (on-policy, bound to C_k, world options = the VM training worlds 16 players/128x64/
# separation 9/movement for collect AND both anchors).
#
# Resumable at stage granularity (Spot preemption, JOB_TIMEOUT, or a rerun): each stage ends
# by writing an atomic done-marker <stage>.done.json with its input checkpoint sha256 and its
# output files' sha256; a stage whose marker verifies is skipped. Markers and payloads are
# uploaded to DEST/iter-NNN/ (payloads content-addressed as <sha16>-<name>, marker last), so a
# lost local disk is restored from GCS. The service account cannot overwrite objects: every
# object name is written once. EXIT_STATUS ("0 <utc> ...") is uploaded once, last, only when
# all ITERATIONS are done; a failure uploads FAIL-<launch> instead (the run stays resumable;
# rerun the same line after fixing the cause). A TERM (VM stop or the JOB_TIMEOUT cap) is an
# interruption: no status, rerun the same line to continue.
#
# Conventions as ops/r5/step-check/collect_job.sh: code tarball (git archive, commit-id check),
# shared venv and bridge (sha256 99131dce...), OMP/MKL 1 thread, nice 10, flock + pgrep guards,
# launch-stamped logs. It NEVER stops the VM, touches no crontab and no other job's WORK/DEST.
#
# Required env: SHA (the pilot code commit; its tarball uploaded to CODE_GCS first).
# Optional (defaults below): PILOT_ID ITERATIONS TURNS ROLLOUT_WORLDS TRAIN_ANCHOR_WORLDS
#   GATE_WORLDS GATE_TELEMETRY PILOT_BYPASS ANCHOR_JOBS ANCHOR_THREADS ANCHOR_PREFIX_TERMS
#   ANCHOR_TAIL_DRAWS ANCHOR_TAIL_RHO ANCHOR_TAIL_CAP ANCHOR_PATHS OPTIMIZER_SEED
#   ACTOR_ENCODER_LR ACTOR_HEAD_LR ACTOR_LR_WARMUP_STEPS TARGET_KL EPOCHS BATCH_SIZE CRITIC_LR
#   CRITIC_EPOCHS ANCHOR_WEIGHT START_ITERATION JOB_TIMEOUT MIN_MEM_GB GCS_ROOT CODE_GCS
#   START_CKPT_GCS (alias CKPT_GCS) CKPT_SHA256 HEAD_ENTROPY_FLOOR DEST WORK CODEROOT PY BRIDGE
#   BRIDGE_SHA256 PLAYERS WIDTH HEIGHT SEPARATION
# Exploration knobs: START_CKPT_GCS + CKPT_SHA256 name a (e.g. labor_head_temper) start
# checkpoint; HEAD_ENTROPY_FLOOR='head=H_min:w[,...]' is passed to train --head-entropy-floor
# every iteration (empty: off, train unchanged). The default PILOT_ID gains -s<start sha8>
# and -f<floor sha8> suffixes for them, so the D7/no-floor default id stays pilot-<sha7>.
#
# Rollout length: TURNS (default 8; the next run sets TURNS=16 explicitly). No stage has its
# own collect budget; only the whole-job cap does, and its default scales with TURNS: 7h at
# TURNS<=8, else ceil(7h x TURNS/8) (14h at 16: camp counts roughly double by turn 8, so
# later turns cost more; the critical path is still the anchors, which do not depend on
# TURNS). An explicit JOB_TIMEOUT always wins. A TERM at the cap is an interruption: rerun.
#
# WP-B paired branches (labor_branch_collect; default BRANCH_SELECT=off: collect unchanged):
#   BRANCH_SELECT off|uniform|adaptive -> collect --branch-select; and, each only when set:
#   BRANCH_RATE (--branch-rate), BRANCH_FLOOR (--branch-floor), BRANCH_HORIZON 2|3
#   (--branch-horizon), BRANCH_CONTINUATION actor|h12, BRANCH_SHADOW 1|0 (--branch-shadow /
#   --no-branch-shadow; unset = only when needed), BRANCH_STREAMS, BRANCH_SEED.
#   With branching on, the collect stage also writes rollout.branches.pt and
#   branch-report.json (labor_branch_report, bound to rollout.pt by sha256): both are collect
#   outputs (marker + upload) and the summary embeds the report as "branches". The default
#   PILOT_ID gains -b<sha8 of the branch spec>. Branch cost is NOT yet measured: with
#   branching on, JOB_TIMEOUT must be given explicitly (the TURNS-scaled default is refused).
#
# Launch (detached; rerun the same line to resume):
#   setsid nohup bash $HOME/ppo_pilot_job.sh > $HOME/r5/ppo-pilot.out 2>&1 < /dev/null &
set -euo pipefail

case "${BRANCH_SELECT:-off}" in off|uniform|adaptive) ;; *) echo "BRANCH_SELECT must be off, uniform or adaptive" >&2; exit 2 ;; esac
if [ "${BRANCH_SELECT:-off}" != "off" ] && [ -z "${JOB_TIMEOUT:-}" ]; then
  echo "BRANCH_SELECT=${BRANCH_SELECT} needs an explicit JOB_TIMEOUT (branch cost is not measured yet)" >&2; exit 2
fi
if [ -z "${JOB_TIMEOUT:-}" ]; then
  case "${TURNS:-8}" in ''|*[!0-9]*) echo "TURNS must be a positive integer" >&2; exit 2 ;; esac
  if [ "${TURNS:-8}" -le 8 ]; then JOB_TIMEOUT=7h; else JOB_TIMEOUT="$(( (7 * TURNS + 7) / 8 ))h"; fi
fi
if [ "${PILOT_UNDER_TIMEOUT:-0}" != "1" ]; then
  export PILOT_UNDER_TIMEOUT=1
  exec timeout --signal=TERM --kill-after=180 "$JOB_TIMEOUT" bash "$0" "$@"
fi

SHA="${SHA:-__FILL_PILOT_SHA__}"
case "$SHA" in *__*) echo "unfilled SHA (export SHA=<pilot code commit>)" >&2; exit 2 ;; esac
GCS_ROOT="${GCS_ROOT:-gs://anthropocene-rl-datasets-508912/model9/r5}"
CODE_GCS="${CODE_GCS:-$GCS_ROOT/code/rl-pilot-${SHA:0:7}.tar.gz}"
D7_START_SHA256=14e749b0a50415f9f1b95d306bf3d8c19a810a29ef0f2755181d856566ba6483
# START_CKPT_GCS: the iteration-0 input checkpoint (CKPT_GCS is its historical name); a
# tempered start (labor_head_temper) is uploaded separately and named here, with its sha256
# in CKPT_SHA256 (the job refuses a checkpoint whose sha256 differs).
START_CKPT_GCS="${START_CKPT_GCS:-${CKPT_GCS:-$GCS_ROOT/d7-run/checkpoint-start.pt}}"
CKPT_SHA256="${CKPT_SHA256:-$D7_START_SHA256}"
# HEAD_ENTROPY_FLOOR: passed to train --head-entropy-floor ('head=H_min:w[,...]'); empty = off.
HEAD_ENTROPY_FLOOR="${HEAD_ENTROPY_FLOOR:-}"
# WP-B branch knobs (see the header); empty values keep collect's defaults.
BRANCH_SELECT="${BRANCH_SELECT:-off}"
BRANCH_RATE="${BRANCH_RATE:-}"; BRANCH_FLOOR="${BRANCH_FLOOR:-}"; BRANCH_HORIZON="${BRANCH_HORIZON:-}"
BRANCH_CONTINUATION="${BRANCH_CONTINUATION:-}"; BRANCH_SHADOW="${BRANCH_SHADOW:-}"
BRANCH_STREAMS="${BRANCH_STREAMS:-}"; BRANCH_SEED="${BRANCH_SEED:-}"
case "$BRANCH_SHADOW" in ''|0|1) ;; *) echo "BRANCH_SHADOW must be empty, 0 or 1" >&2; exit 2 ;; esac
BRANCH_SPEC="off"
if [ "$BRANCH_SELECT" != "off" ]; then
  BRANCH_SPEC="select=$BRANCH_SELECT rate=$BRANCH_RATE floor=$BRANCH_FLOOR horizon=$BRANCH_HORIZON continuation=$BRANCH_CONTINUATION shadow=$BRANCH_SHADOW streams=$BRANCH_STREAMS seed=$BRANCH_SEED"
fi
# PILOT_ID: pilot-<sha7>, plus -s<start sha8> for a non-D7 start and -f<sha8 of the floor
# spec> with a floor (and -b<sha8 of the branch spec> with branching), so a different start,
# floor or branch setting never resumes another run's markers.
if [ -z "${PILOT_ID:-}" ]; then
  PILOT_ID="pilot-${SHA:0:7}"
  if [ "$CKPT_SHA256" != "$D7_START_SHA256" ]; then PILOT_ID="$PILOT_ID-s${CKPT_SHA256:0:8}"; fi
  if [ -n "$HEAD_ENTROPY_FLOOR" ]; then
    PILOT_ID="$PILOT_ID-f$(printf %s "$HEAD_ENTROPY_FLOOR" | sha256sum | cut -c1-8)"
  fi
  if [ "$BRANCH_SELECT" != "off" ]; then
    PILOT_ID="$PILOT_ID-b$(printf %s "$BRANCH_SPEC" | sha256sum | cut -c1-8)"
  fi
fi
DEST="${DEST:-$GCS_ROOT/ppo-pilot/$PILOT_ID}"
WORK="${WORK:-$HOME/r5/ppo-pilot/$PILOT_ID}"
CODEROOT="${CODEROOT:-$HOME/code/rl-pilot-${SHA:0:7}}"
CODE="$CODEROOT/rl"
PY="${PY:-$HOME/m9-bench/venv/bin/python}"
BRIDGE="${BRIDGE:-$HOME/code/r5s841c38e/rl/integration/bin/model9/Anthropocene.Model9Bridge.dll}"
BRIDGE_SHA256="${BRIDGE_SHA256:-99131dce9c7f0d0d66988b05c3c3e9adea6f6ad3407ce4cd0366d6a14b25eb6b}"
ITERATIONS="${ITERATIONS:-3}"
TURNS="${TURNS:-8}"
ROLLOUT_WORLDS="${ROLLOUT_WORLDS:-3}"
TRAIN_ANCHOR_WORLDS="${TRAIN_ANCHOR_WORLDS:-$ROLLOUT_WORLDS}"
GATE_WORLDS="${GATE_WORLDS:-2}"
GATE_TELEMETRY="${GATE_TELEMETRY:-1}"
PILOT_BYPASS="${PILOT_BYPASS:-1}"
ANCHOR_JOBS="${ANCHOR_JOBS:-0}"          # per producer; 0 = one process per (world, path) unit
ANCHOR_THREADS="${ANCHOR_THREADS:-1}"
ANCHOR_PREFIX_TERMS="${ANCHOR_PREFIX_TERMS:-}"   # empty: the producer's defaults (25/16/0.95/200/4)
ANCHOR_TAIL_DRAWS="${ANCHOR_TAIL_DRAWS:-}"
ANCHOR_TAIL_RHO="${ANCHOR_TAIL_RHO:-}"
ANCHOR_TAIL_CAP="${ANCHOR_TAIL_CAP:-}"
ANCHOR_PATHS="${ANCHOR_PATHS:-}"
OPTIMIZER_SEED="${OPTIMIZER_SEED:-0}"
ACTOR_ENCODER_LR="${ACTOR_ENCODER_LR:-1e-7}"
ACTOR_HEAD_LR="${ACTOR_HEAD_LR:-3e-5}"
ACTOR_LR_WARMUP_STEPS="${ACTOR_LR_WARMUP_STEPS:-12}"
TARGET_KL="${TARGET_KL:-0.02}"
EPOCHS="${EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
CRITIC_LR="${CRITIC_LR:-3e-4}"
CRITIC_EPOCHS="${CRITIC_EPOCHS:-1}"
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-1.0}"
START_ITERATION="${START_ITERATION:-0}"
PLAYERS="${PLAYERS:-16}"; WIDTH="${WIDTH:-128}"; HEIGHT="${HEIGHT:-64}"; SEPARATION="${SEPARATION:-9}"
MIN_MEM_GB="${MIN_MEM_GB:-16}"
case "$GATE_TELEMETRY$PILOT_BYPASS" in 00) echo "GATE_TELEMETRY=0 needs PILOT_BYPASS=1 (no report: the actor would never update)" >&2; exit 2 ;; esac
case "$GATE_TELEMETRY" in 0|1) ;; *) echo "GATE_TELEMETRY must be 0 or 1" >&2; exit 2 ;; esac
case "$PILOT_BYPASS" in 0|1) ;; *) echo "PILOT_BYPASS must be 0 or 1" >&2; exit 2 ;; esac

ts() { date -u +%FT%TZ; }
LAUNCH=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$WORK/logs" "$WORK/ckpt"
LOG_FILE="$WORK/logs/job-$LAUNCH.log"
CUR_STAGE=startup
log() { echo "$(ts) [ppo-pilot] $*" | tee -a "$LOG_FILE" >&2; }
die() { log "FAIL[$CUR_STAGE] $*"; exit 1; }
sha256_of() { sha256sum "$1" | cut -d' ' -f1; }
gcs_put() { timeout 900 gcloud storage cp --no-user-output-enabled "$1" "$2" < /dev/null || { log "upload failed: $1 -> $2"; return 1; }; }
gcs_get() { timeout 900 gcloud storage cp --no-user-output-enabled "$1" "$2" < /dev/null; }
remote_has() { timeout 120 gcloud storage ls "$1" > /dev/null 2>&1 < /dev/null; }
put_once() { remote_has "$2" || gcs_put "$1" "$2"; }   # object names are written once
mem_available_gb() { awk '/MemAvailable/{a=$2} /MemFree/{f=$2} END{print int((a ? a : f) / 1048576)}' /proc/meminfo; }
pyrun() { (cd "$CODE" && "$PY" "$@"); }

exec 9> "$WORK/job.lock"
flock -n 9 || { log "another ppo_pilot_job.sh holds $WORK/job.lock; exiting"; exit 3; }
if pgrep -f "[a]nthropocene_rl.autonomous_v7.labor_.* .*$WORK/" > /dev/null; then
  log "a pilot stage is already running: $(pgrep -af "[a]nthropocene_rl.autonomous_v7.labor_.* .*$WORK/" | head -2); exiting"
  exit 3
fi
if remote_has "$DEST/EXIT_STATUS"; then
  log "$DEST/EXIT_STATUS already exists; the pilot is complete (use a new PILOT_ID/DEST to rerun)"
  exit 0
fi

SKIP_STATUS=0
BG_PIDS=""
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
stop_background() {
  local pid all=""
  for pid in $BG_PIDS; do all="$all $(tree_pids "$pid")"; done
  [ -n "${all// }" ] && kill -TERM $all 2> /dev/null
  return 0
}
finish() {
  local status=$?
  trap - EXIT
  set +e
  stop_background
  if [ "$SKIP_STATUS" != "1" ] && [ "$status" -ne 0 ]; then
    echo "FAIL:$CUR_STAGE $(ts)" > "$WORK/FAIL-$LAUNCH"
    gcs_put "$WORK/FAIL-$LAUNCH" "$DEST/FAIL-$LAUNCH"
  fi
  gcs_put "$LOG_FILE" "$DEST/logs/job-$LAUNCH.log"
  if [ "$status" -eq 0 ] && [ -f "$WORK/EXIT_STATUS" ]; then
    if remote_has "$DEST/EXIT_STATUS"; then
      log "$DEST/EXIT_STATUS appeared during this launch; not overwriting"
    else
      gcs_put "$WORK/EXIT_STATUS" "$DEST/EXIT_STATUS"   # last and once; this job never stops the VM
    fi
  fi
  exit "$status"
}
trap finish EXIT
trap 'SKIP_STATUS=1; log "signal received: interrupted (VM stop or JOB_TIMEOUT), no status; rerun the same line to resume"; exit 143' TERM INT HUP

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
    gcs_get "$CODE_GCS" "$CODEROOT.tmp/code.tar.gz" || die "download failed: $CODE_GCS (build and upload the tarball first)"
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
  local where
  where=$(pyrun -c 'import anthropocene_rl; print(anthropocene_rl.__file__)' 2>&1) || die "cannot import anthropocene_rl: $where"
  case "$where" in "$CODE"/src/*) ;; *) die "anthropocene_rl resolves to $where, not $CODE/src" ;; esac
  case "$(pyrun -m anthropocene_rl.autonomous_v7.labor_training train --help < /dev/null 2>&1 || true)" in
    *--pilot-ungated-actor*) ;; *) die "code $SOURCE_COMMIT lacks train --pilot-ungated-actor" ;; esac
  case "$(pyrun -m anthropocene_rl.autonomous_v7.labor_anchor_producer --help < /dev/null 2>&1 || true)" in
    *--jobs*--threads*) ;; *) die "code $SOURCE_COMMIT lacks labor_anchor_producer --jobs/--threads" ;; esac
  if [ -n "$HEAD_ENTROPY_FLOOR" ]; then
    case "$(pyrun -m anthropocene_rl.autonomous_v7.labor_training train --help < /dev/null 2>&1 || true)" in
      *--head-entropy-floor*) ;; *) die "code $SOURCE_COMMIT lacks train --head-entropy-floor" ;; esac
    pyrun -c 'import sys; from anthropocene_rl.autonomous_v7.labor_ppo import parse_head_entropy_floor as p; print(p(sys.argv[1]))' "$HEAD_ENTROPY_FLOOR" > /dev/null \
      || die "HEAD_ENTROPY_FLOOR '$HEAD_ENTROPY_FLOOR' does not parse"
  fi
  if [ "$BRANCH_SELECT" != "off" ]; then
    case "$(pyrun -m anthropocene_rl.autonomous_v7.labor_training collect --help < /dev/null 2>&1 || true)" in
      *--branch-select*) ;; *) die "code $SOURCE_COMMIT lacks collect --branch-select" ;; esac
    case "$(pyrun -m anthropocene_rl.autonomous_v7.labor_pilot summary --help < /dev/null 2>&1 || true)" in
      *--branch-report*) ;; *) die "code $SOURCE_COMMIT lacks labor_pilot summary --branch-report" ;; esac
  fi
  pyrun -m anthropocene_rl.autonomous_v7.labor_pilot plan --iteration 0 > /dev/null || die "code $SOURCE_COMMIT lacks labor_pilot"
  log "code $SOURCE_COMMIT; bridge ${BRIDGE_SHA256:0:16}; free mem $(mem_available_gb) GB; free disk $(df -BG --output=avail "$WORK" | tail -1 | tr -d ' '); cpus $(getconf _NPROCESSORS_ONLN)"
}

stage_start_checkpoint() {
  CUR_STAGE=start-checkpoint
  START_CKPT="$WORK/ckpt/checkpoint-start.pt"
  if [ ! -f "$START_CKPT" ] || [ "$(sha256_of "$START_CKPT")" != "$CKPT_SHA256" ]; then
    rm -f "$START_CKPT"
    gcs_get "$START_CKPT_GCS" "$START_CKPT" || die "cannot fetch $START_CKPT_GCS"
  fi
  [ "$(sha256_of "$START_CKPT")" = "$CKPT_SHA256" ] || die "start checkpoint sha256 != $CKPT_SHA256"
  # Stage markers bind input checkpoints, not train flags: a rerun of this WORK with another
  # floor (possible only with an explicit PILOT_ID) would skip finished train stages.
  local explore="start=$CKPT_SHA256 floor=${HEAD_ENTROPY_FLOOR:-off}"
  [ "$BRANCH_SELECT" != "off" ] && explore="$explore branches=[$BRANCH_SPEC]"
  if [ -f "$WORK/explore-config" ] && [ "$(cat "$WORK/explore-config")" != "$explore" ]; then
    die "this WORK ran with '$(cat "$WORK/explore-config")', not '$explore' (use a new PILOT_ID)"
  fi
  echo "$explore" > "$WORK/explore-config"
  log "start checkpoint ${CKPT_SHA256:0:16} from $START_CKPT_GCS; head tempering: $(pyrun -c 'import json, sys, torch; s = torch.load(sys.argv[1], map_location="cpu", weights_only=True); print(json.dumps([{k: r.get(k) for k in ("targets", "transform")} for r in s.get("headTempering") or []], default=str))' "$START_CKPT" 2>&1 | tail -1)"
}

# --- stage markers ------------------------------------------------------------------------
# marker: {"stage", "iteration", "inputCheckpointSha256", "outputs": {name: sha256}, "seconds",
#          "launch", "completedUtc"}; outputs are files in ITER_DIR.
write_marker() {  # STAGE SECONDS NAME...
  local stage=$1 seconds=$2; shift 2
  pyrun - "$ITER_DIR" "$stage" "$ITER" "$IN_SHA" "$seconds" "$LAUNCH" "$@" <<'PY' || return 1
import hashlib, json, os, sys, time
directory, stage, iteration, input_sha, seconds, launch, *names = sys.argv[1:]
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
record = {"stage": stage, "iteration": int(iteration), "inputCheckpointSha256": input_sha,
          "outputs": {name: sha(os.path.join(directory, name)) for name in names},
          "seconds": float(seconds), "launch": launch,
          "completedUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
path = os.path.join(directory, f"{stage}.done.json")
with open(path + ".tmp", "w") as handle:
    json.dump(record, handle, sort_keys=True)
os.replace(path + ".tmp", path)
PY
}
marker_ok() {  # STAGE -> 0 when the marker matches the current input and every output verifies
  [ -f "$ITER_DIR/$1.done.json" ] || return 1
  pyrun - "$ITER_DIR" "$1" "$IN_SHA" <<'PY' > /dev/null 2>&1
import hashlib, json, os, sys
directory, stage, input_sha = sys.argv[1:]
record = json.load(open(os.path.join(directory, f"{stage}.done.json")))
assert record["stage"] == stage and record["inputCheckpointSha256"] == input_sha
for name, digest in record["outputs"].items():
    h = hashlib.sha256()
    with open(os.path.join(directory, name), "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    assert h.hexdigest() == digest, name
PY
}
marker_outputs() { pyrun -c 'import json,sys; [print(n, s) for n, s in json.load(open(sys.argv[1]))["outputs"].items()]' "$ITER_DIR/$1.done.json"; }
upload_stage() {  # STAGE: payloads (content-addressed) first, the marker last
  local name sha remote="$ITER_DIR/.remote-$1.json"
  while read -r name sha; do
    put_once "$ITER_DIR/$name" "$ITER_DEST/${sha:0:16}-$name" || return 1
  done < <(marker_outputs "$1")
  if remote_has "$ITER_DEST/$1.done.json"; then
    # Written once: an existing remote marker for other outputs (a stage rerun after its
    # input changed) is kept; this launch's marker goes next to it under a stamped name.
    gcs_get "$ITER_DEST/$1.done.json" "$remote" || return 1
    if ! pyrun -c 'import json,sys; a,b=(json.load(open(p)) for p in sys.argv[1:]); sys.exit(0 if (a["inputCheckpointSha256"],a["outputs"])==(b["inputCheckpointSha256"],b["outputs"]) else 1)' "$remote" "$ITER_DIR/$1.done.json"; then
      log "WARN iter $ITER $1: remote marker differs (stale); uploading $1.done-$LAUNCH.json"
      put_once "$ITER_DIR/$1.done.json" "$ITER_DEST/$1.done-$LAUNCH.json"
    fi
    rm -f "$remote"
    return 0
  fi
  gcs_put "$ITER_DIR/$1.done.json" "$ITER_DEST/$1.done.json"
}
restore_stage() {  # STAGE: rebuild a stage from GCS when its marker was uploaded
  remote_has "$ITER_DEST/$1.done.json" || return 1
  local name sha tmp="$ITER_DIR/.restore-$1.json"
  gcs_get "$ITER_DEST/$1.done.json" "$tmp" || return 1
  while read -r name sha; do
    gcs_get "$ITER_DEST/${sha:0:16}-$name" "$ITER_DIR/$name" || return 1
  done < <(pyrun -c 'import json,sys; [print(n, s) for n, s in json.load(open(sys.argv[1]))["outputs"].items()]' "$tmp")
  mv "$tmp" "$ITER_DIR/$1.done.json"
  marker_ok "$1"
}
run_stage() {  # STAGE FUNCTION: skip / restore / run + mark + upload
  local stage=$1 fn=$2 start
  if marker_ok "$stage"; then
    log "iter $ITER $stage: done (marker verifies); skipped"
    upload_stage "$stage" || die "upload of $stage failed"
    return 0
  fi
  if restore_stage "$stage"; then
    log "iter $ITER $stage: restored from $ITER_DEST"
    return 0
  fi
  rm -f "$ITER_DIR/$stage.done.json"
  start=$(date +%s)
  log "iter $ITER $stage: start (mem $(mem_available_gb) GB, load $(cut -d' ' -f1 /proc/loadavg))"
  "$fn" || die "iter $ITER $stage failed (log $ITER_DIR/logs/$stage-$LAUNCH.log)"
  local outputs; outputs=$(cat "$ITER_DIR/.outputs-$stage")
  # shellcheck disable=SC2086
  write_marker "$stage" "$(( $(date +%s) - start ))" $outputs || die "iter $ITER $stage: marker failed"
  log "iter $ITER $stage: done in $(( $(date +%s) - start ))s"
  upload_stage "$stage" || die "upload of $stage failed"
  put_once "$ITER_DIR/logs/$stage-$LAUNCH.log" "$ITER_DEST/logs/$stage-$LAUNCH.log" || true
}

# --- stages (each writes its outputs into ITER_DIR and lists them in .outputs-<stage>) -----
world_flags=()
anchor_flags=()
branch_flags=()
init_flags() {
  world_flags=(--players "$PLAYERS" --width "$WIDTH" --height "$HEIGHT" --separation "$SEPARATION")
  anchor_flags=(--optimizer-seed "$OPTIMIZER_SEED" --allow-vm-scale --threads "$ANCHOR_THREADS" --progress)
  [ -n "$ANCHOR_PREFIX_TERMS" ] && anchor_flags+=(--prefix-terms "$ANCHOR_PREFIX_TERMS")
  [ -n "$ANCHOR_TAIL_DRAWS" ] && anchor_flags+=(--tail-draws "$ANCHOR_TAIL_DRAWS")
  [ -n "$ANCHOR_TAIL_RHO" ] && anchor_flags+=(--tail-rho "$ANCHOR_TAIL_RHO")
  [ -n "$ANCHOR_TAIL_CAP" ] && anchor_flags+=(--tail-cap "$ANCHOR_TAIL_CAP")
  [ -n "$ANCHOR_PATHS" ] && anchor_flags+=(--paths "$ANCHOR_PATHS")
  branch_flags=()
  if [ "$BRANCH_SELECT" != "off" ]; then
    branch_flags=(--branch-select "$BRANCH_SELECT")
    [ -n "$BRANCH_RATE" ] && branch_flags+=(--branch-rate "$BRANCH_RATE")
    [ -n "$BRANCH_FLOOR" ] && branch_flags+=(--branch-floor "$BRANCH_FLOOR")
    [ -n "$BRANCH_HORIZON" ] && branch_flags+=(--branch-horizon "$BRANCH_HORIZON")
    [ -n "$BRANCH_CONTINUATION" ] && branch_flags+=(--branch-continuation "$BRANCH_CONTINUATION")
    [ "$BRANCH_SHADOW" = "1" ] && branch_flags+=(--branch-shadow)
    [ "$BRANCH_SHADOW" = "0" ] && branch_flags+=(--no-branch-shadow)
    [ -n "$BRANCH_STREAMS" ] && branch_flags+=(--branch-streams "$BRANCH_STREAMS")
    [ -n "$BRANCH_SEED" ] && branch_flags+=(--branch-seed "$BRANCH_SEED")
  fi
  return 0
}
plan_value() { pyrun -c 'import json,sys; v=json.load(open(sys.argv[1]))[sys.argv[2]]; print(",".join(map(str,v)) if isinstance(v,list) else v)' "$ITER_DIR/plan.json" "$1"; }
anchor_jobs() {  # WORLD_CSV -> --jobs value
  local worlds=$1 paths="${ANCHOR_PATHS:-4}" units
  units=$(( $(echo "$worlds" | tr ',' '\n' | grep -c .) * paths ))
  if [ "$ANCHOR_JOBS" -gt 0 ] && [ "$ANCHOR_JOBS" -lt "$units" ]; then echo "$ANCHOR_JOBS"; else echo "$units"; fi
}

do_collect() {
  local worlds; worlds=$(plan_value rolloutWorlds)
  rm -f "$ITER_DIR/rollout.pt" "$ITER_DIR/rollout.branches.pt" "$ITER_DIR/branch-report.json"
  nice -n 10 "$PY" -m anthropocene_rl.autonomous_v7.labor_training collect \
    --checkpoint "$IN_CKPT" --worlds "$worlds" --turns "$TURNS" --allow-vm-scale \
    "${world_flags[@]}" --movement "${branch_flags[@]}" --out "$ITER_DIR/rollout.pt" \
    >> "$ITER_DIR/logs/collect-$LAUNCH.log" 2>&1 < /dev/null || return 1
  "$PY" -m anthropocene_rl.autonomous_v7.labor_pilot check-rollout --rollout "$ITER_DIR/rollout.pt" \
    --checkpoint "$IN_CKPT" --worlds "$worlds" --turns "$TURNS" "${world_flags[@]}" \
    > "$ITER_DIR/rollout.check.json" 2>> "$ITER_DIR/logs/collect-$LAUNCH.log" || return 1
  if [ "$BRANCH_SELECT" != "off" ]; then
    "$PY" -m anthropocene_rl.autonomous_v7.labor_branch_report --branches "$ITER_DIR/rollout.branches.pt" \
      --rollout "$ITER_DIR/rollout.pt" --out "$ITER_DIR/branch-report.json" \
      2>> "$ITER_DIR/logs/collect-$LAUNCH.log" || return 1
    echo "rollout.pt rollout.check.json rollout.branches.pt branch-report.json" > "$ITER_DIR/.outputs-collect"
  else
    echo "rollout.pt rollout.check.json" > "$ITER_DIR/.outputs-collect"
  fi
}
do_anchor() {  # ROLE PLAN_WORLDS_KEY PLAN_SEED_KEY NAME
  local role=$1 worlds base name=$4 log
  worlds=$(plan_value "$2"); base=$(plan_value "$3"); log="$ITER_DIR/logs/$name-$LAUNCH.log"
  rm -f "$ITER_DIR/$name.npz"
  nice -n 10 "$PY" -m anthropocene_rl.autonomous_v7.labor_anchor_producer \
    --checkpoint "$IN_CKPT" --worlds "$worlds" --continuation-seed-base "$base" --role "$role" \
    --jobs "$(anchor_jobs "$worlds")" "${anchor_flags[@]}" "${world_flags[@]}" \
    --out "$ITER_DIR/$name.npz" >> "$log" 2>&1 < /dev/null || return 1
  "$PY" -m anthropocene_rl.autonomous_v7.labor_pilot check-anchor --anchor "$ITER_DIR/$name.npz" \
    --checkpoint "$IN_CKPT" --role "$role" --worlds "$worlds" "${world_flags[@]}" \
    > "$ITER_DIR/$name.check.json" 2>> "$log" || return 1
  echo "$name.npz $name.check.json" > "$ITER_DIR/.outputs-$name"
}
do_train_anchor() { do_anchor train-anchor trainAnchorWorlds trainAnchorSeedBase train-anchor; }
do_actor_gate() { do_anchor actor-gate gateWorlds gateSeedBase actor-gate; }
do_calibration() {
  local log="$ITER_DIR/logs/calibration-$LAUNCH.log"
  rm -f "$ITER_DIR/calibration.json"
  "$PY" -m anthropocene_rl.autonomous_v7.labor_value_calibration --checkpoint "$IN_CKPT" \
    --anchor "$ITER_DIR/actor-gate.npz" --out "$ITER_DIR/calibration.json" >> "$log" 2>&1 < /dev/null || return 1
  "$PY" -m anthropocene_rl.autonomous_v7.labor_pilot check-calibration --report "$ITER_DIR/calibration.json" \
    --checkpoint "$IN_CKPT" --anchor "$ITER_DIR/actor-gate.npz" > "$ITER_DIR/calibration.check.json" 2>> "$log" || return 1
  echo "calibration.json calibration.check.json" > "$ITER_DIR/.outputs-calibration"
}
do_train() {
  local log="$ITER_DIR/logs/train-$LAUNCH.log" extra=()
  [ "$GATE_TELEMETRY" = "1" ] && extra+=(--calibration-report "$ITER_DIR/calibration.json")
  [ "$PILOT_BYPASS" = "1" ] && extra+=(--pilot-ungated-actor)
  [ -n "$HEAD_ENTROPY_FLOOR" ] && extra+=(--head-entropy-floor "$HEAD_ENTROPY_FLOOR")
  rm -f "$ITER_DIR/checkpoint-out.pt"
  nice -n 10 "$PY" -m anthropocene_rl.autonomous_v7.labor_training train \
    --checkpoint "$IN_CKPT" --rollout "$ITER_DIR/rollout.pt" --anchor "$ITER_DIR/train-anchor.npz" \
    --a-mode off --actor-encoder-lr "$ACTOR_ENCODER_LR" --actor-head-lr "$ACTOR_HEAD_LR" \
    --actor-lr-warmup-steps "$ACTOR_LR_WARMUP_STEPS" --target-kl "$TARGET_KL" --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" --critic-lr "$CRITIC_LR" --critic-epochs "$CRITIC_EPOCHS" \
    --anchor-weight "$ANCHOR_WEIGHT" "${extra[@]}" --out "$ITER_DIR/checkpoint-out.pt" \
    >> "$log" 2>&1 < /dev/null || return 1
  echo "checkpoint-out.pt" > "$ITER_DIR/.outputs-train"
}
do_summary() {
  local log="$ITER_DIR/logs/summary-$LAUNCH.log" extra=()
  pyrun - "$ITER_DIR" > "$ITER_DIR/stage-seconds.json" <<'PY' || return 1
import glob, json, os, sys
out = {}
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*.done.json"))):
    record = json.load(open(path))
    out[record["stage"]] = {"seconds": record["seconds"], "launch": record["launch"]}
print(json.dumps(out, sort_keys=True))
PY
  if [ "$GATE_TELEMETRY" = "1" ]; then
    extra+=(--gate-anchor "$ITER_DIR/actor-gate.npz" --calibration "$ITER_DIR/calibration.json")
  fi
  [ "$PILOT_BYPASS" = "1" ] && extra+=(--bypass-expected)
  [ "$BRANCH_SELECT" != "off" ] && extra+=(--branch-report "$ITER_DIR/branch-report.json")
  "$PY" -m anthropocene_rl.autonomous_v7.labor_pilot summary --iteration "$ITER" \
    --checkpoint-in "$IN_CKPT" --checkpoint-out "$ITER_DIR/checkpoint-out.pt" \
    --rollout "$ITER_DIR/rollout.pt" --train-anchor "$ITER_DIR/train-anchor.npz" "${extra[@]}" \
    --stage-seconds "$ITER_DIR/stage-seconds.json" > "$ITER_DIR/summary.json" 2>> "$log" < /dev/null || return 1
  echo "summary.json stage-seconds.json" > "$ITER_DIR/.outputs-summary"
}

stage_parallel() {  # collect, train-anchor and (telemetry) actor-gate, concurrently from C_k
  CUR_STAGE="iter-$ITER-data"
  [ "$(mem_available_gb)" -ge "$MIN_MEM_GB" ] || die "MemAvailable $(mem_available_gb) GB < $MIN_MEM_GB GB"
  local names=(collect train-anchor) fns=(do_collect do_train_anchor) pids=() rcs=() i failed=""
  if [ "$GATE_TELEMETRY" = "1" ]; then names+=(actor-gate); fns+=(do_actor_gate); fi
  for i in "${!names[@]}"; do
    (cd "$CODE" && run_stage "${names[$i]}" "${fns[$i]}") &
    pids+=($!)
  done
  BG_PIDS="${pids[*]}"
  for i in "${!pids[@]}"; do
    rcs[$i]=0; wait "${pids[$i]}" || rcs[$i]=$?
    [ "${rcs[$i]}" -eq 0 ] || failed="$failed ${names[$i]}"
  done
  BG_PIDS=""
  if [ -n "$failed" ]; then CUR_STAGE="iter-$ITER-${failed# }"; die "stage(s)$failed failed"; fi
}

run_iteration() {
  ITER=$1
  ITER_DIR="$WORK/iter-$(printf %03d "$ITER")"
  ITER_DEST="$DEST/iter-$(printf %03d "$ITER")"
  mkdir -p "$ITER_DIR/logs"
  if [ "$ITER" -eq 0 ]; then
    IN_CKPT="$START_CKPT"
  else
    IN_CKPT="$WORK/iter-$(printf %03d $(( ITER - 1 )))/checkpoint-out.pt"
  fi
  [ -f "$IN_CKPT" ] || die "iteration $ITER input checkpoint $IN_CKPT is missing"
  IN_SHA=$(sha256_of "$IN_CKPT")
  if marker_ok iteration || restore_stage iteration; then
    log "iter $ITER: complete (marker verifies); ensuring its output checkpoint"
    run_stage train do_train   # verifies, else restores; never retrains a marked iteration
    return 0
  fi
  CUR_STAGE="iter-$ITER-preflight"
  pyrun -m anthropocene_rl.autonomous_v7.labor_pilot plan --iteration "$ITER" \
    --rollout-worlds "$ROLLOUT_WORLDS" --train-anchor-worlds "$TRAIN_ANCHOR_WORLDS" \
    --gate-worlds "$GATE_WORLDS" > "$ITER_DIR/plan.json" || die "plan failed"
  pyrun -m anthropocene_rl.autonomous_v7.labor_pilot preflight --checkpoint "$IN_CKPT" \
    --plan "$ITER_DIR/plan.json" --expected-iteration $(( START_ITERATION + ITER )) \
    --anchor-paths "${ANCHOR_PATHS:-4}" > "$ITER_DIR/preflight.json" 2>> "$LOG_FILE" \
    || die "preflight refused iteration $ITER (see job log)"
  log "iter $ITER: input ${IN_SHA:0:16}; plan $(tr -d '\n ' < "$ITER_DIR/plan.json")"
  stage_parallel
  if [ "$GATE_TELEMETRY" = "1" ]; then CUR_STAGE="iter-$ITER-calibration"; run_stage calibration do_calibration; fi
  CUR_STAGE="iter-$ITER-train"; run_stage train do_train
  CUR_STAGE="iter-$ITER-summary"; run_stage summary do_summary
  log "iter $ITER summary: $(pyrun -c 'import json,sys; s=json.load(open(sys.argv[1])); a=s["actor"]; g=s["gate"] or {}; print({"gateMode": s["actorGateMode"], "gateOpen": g.get("actorGateOpen"), "failing": g.get("failingHeads"), "accepted": a["acceptedSteps"], "attempted": a["attemptedSteps"], "criticLoss": s["critic"]["loss"], "dlogpJoint": s["meanAbsDeltaLogp"].get("initialJoint")})' "$ITER_DIR/summary.json")"
  CUR_STAGE="iter-$ITER-mark"
  echo "summary.json checkpoint-out.pt" > "$ITER_DIR/.outputs-iteration"
  write_marker iteration 0 summary.json checkpoint-out.pt || die "iteration marker failed"
  upload_stage iteration || die "iteration marker upload failed"
}

log "start launch=$LAUNCH pilot=$PILOT_ID sha=$SHA start=${CKPT_SHA256:0:16} floor=${HEAD_ENTROPY_FLOOR:-off} iterations=$ITERATIONS turns=$TURNS worlds rollout=$ROLLOUT_WORLDS anchor=$TRAIN_ANCHOR_WORLDS gate=$GATE_WORLDS telemetry=$GATE_TELEMETRY bypass=$PILOT_BYPASS anchor_jobs=$ANCHOR_JOBS threads=$ANCHOR_THREADS reset=$PLAYERS/${WIDTH}x$HEIGHT/sep$SEPARATION/movement branches=[$BRANCH_SPEC] dest=$DEST cap=$JOB_TIMEOUT"
rm -f "$WORK/EXIT_STATUS"
stage_code
stage_start_checkpoint
init_flags
for (( K = 0; K < ITERATIONS; K++ )); do
  run_iteration "$K"
done
CUR_STAGE=final
FINAL="$WORK/iter-$(printf %03d $(( ITERATIONS - 1 )))/checkpoint-out.pt"
echo "0 $(ts) pilot=$PILOT_ID iterations=$ITERATIONS source=$SOURCE_COMMIT start=${CKPT_SHA256:0:16} final=$(sha256_of "$FINAL" | cut -c1-16) bypass=$PILOT_BYPASS" > "$WORK/EXIT_STATUS"
log "done; EXIT_STATUS 0 (uploaded last by the exit handler); the VM is left running"
