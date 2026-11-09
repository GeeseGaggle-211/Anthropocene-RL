#!/usr/bin/env bash
# r5 actor step check, iteration-2 collect (CPU VM anthropocene-trainer-60, c3d-highcpu-60
# Spot, us-central1-c). Adapted from ops/r5/d7/d7_collect_job.sh; only the inputs, the
# output folder, the 1 h cap and the checkpoint preflight differ.
#
# One process: labor_training collect from checkpoint-1.pt, the output of ONE real actor-only
# PPO update (labor_a_measure --apply: train --a-mode off's update_labor_policy with rollback
# active, target KL 0.02, encoder lr 1e-7, head lr 3e-5, warm-up 12 Adam steps) applied
# locally to the D7 start checkpoint (sha 14e749b0...) on the D7 rollout 91000-91002 (sha
# d3c75bba...). The critic (task/costs) and its optimizer were NOT updated: they are the start
# checkpoint's, bitwise (critic digest ed0758ed258eb100...). Worlds 91003-91005 (fresh: not in
# any critic-training, A-fit/eval, D7, sealed 3000-3059 or reserved 90154-90159 set), 8
# decisions per world. The rollout's behaviour-policy digest must be checkpoint-1's actor
# digest (BEHAVIOR_DIGEST below; checked before and after the collect).
#
# Why ONE multi-world collect and not one collect per world: `train` and labor_a_measure consume
# exactly one rollout file (one batch, one sourceCheckpointSha256, one sampling-generator
# before/after); there is no batch-merge path in train. Single-world collects would each start
# from the checkpoint's identical sampling-generator state (the same noise stream reused on
# every world) and would need a merge that train never does. The collect passes
# --allow-vm-scale (it only lifts the LOCAL_WORLD_LIMIT 1..3 world-count cap; 3 worlds are
# within it; the reserved/sealed guards still apply). Worlds run sequentially in the process.
#
# World configuration: the training worlds the critic, A and the F1 actor were fitted on,
# read from the fit-h12 search source.json reset_options ({players 16, width 128, height 64,
# separation 9, movement true, withheld 0}) and the panel/DAgger configs (16 / 128x64 / 9, no
# --no-movement): collect --players 16 --width 128 --height 64 --separation 9 --movement.
# Without these flags collect keeps its historical bridge.reset(seed) (8 players, movement
# off, the Small world: a mechanical fixture). The rollout records them (resetOptions) and
# the job checks them after the collect.
#
# Conventions follow actor_diag_job.sh / actor_panel_job.sh / reserved_succ_job.sh: code tarball
# (git archive) with commit-id check and best-effort snapshot commit, the shared venv, the shared
# bridge (sha256 99131dce..., ~/code/r5s841c38e/...; no rebuild: integration/ and every
# bridge-facing module are unchanged from 841c38e/115cab5 to 64bf8a5), ANTHROPOCENE_DOTNET and
# ANTHROPOCENE_MODEL9_BRIDGE exported, OMP/MKL 1 thread, nice 10, flock + pgrep duplicate guards,
# launch-stamped log names (the VM service account cannot overwrite or delete objects),
# EXIT_STATUS uploaded once and LAST ("0 <utc> ..." or "FAIL:<stage> <utc>").
#
# Shares the VM with the actor team (idle queue runner.py, maybe their 12-slot panel). This job
# NEVER stops the VM (shutdown is left to the 8 h guardrail and the actor team's chain), never
# touches the crontab, ~/r5/scaleout, or any other job's WORK/DEST, and kills nothing.
# Hard wall-clock cap: the whole job re-execs itself under `timeout` (JOB_TIMEOUT, default 1h);
# the collect gets the budget minus UPLOAD_RESERVE_S and reports FAIL:timeout when it hits it.
#
# Required env: none (defaults below). Optional: SHA CODE_GCS CKPT_GCS CKPT_SHA256
#   BEHAVIOR_DIGEST DEST WORK CODEROOT PY BRIDGE BRIDGE_SHA256 WORLDS TURNS PLAYERS WIDTH HEIGHT
#   SEPARATION MOVEMENT JOB_TIMEOUT UPLOAD_RESERVE_S MIN_MEM_GB
#
# Launch (detached; rerun the same line after a VM stop: a present rollout is not recollected):
#   setsid nohup bash $HOME/step_check_collect_job.sh > $HOME/r5/step-check-run.out 2>&1 < /dev/null &
set -euo pipefail

JOB_TIMEOUT="${JOB_TIMEOUT:-1h}"
if [ "${STEPCHECK_UNDER_TIMEOUT:-0}" != "1" ]; then
  export STEPCHECK_UNDER_TIMEOUT=1
  exec timeout --signal=TERM --kill-after=120 "$JOB_TIMEOUT" bash "$0" "$@"
fi

SHA="${SHA:-3a178ea868f22f037274e19027a0530b775ad505}"
GCS_ROOT="${GCS_ROOT:-gs://anthropocene-rl-datasets-508912/model9/r5}"
CODE_GCS="${CODE_GCS:-$GCS_ROOT/code/rl-awire-${SHA:0:7}.tar.gz}"
CKPT_GCS="${CKPT_GCS:-$GCS_ROOT/step-check/checkpoint-1.pt}"
CKPT_SHA256="${CKPT_SHA256:-ed99305d59858260ee48fd1d1ae9a4ffdab42e1700f1565f2cab73d89342d465}"
# checkpoint-1's behaviour-policy digest (labor_a_measure --apply report: output.behaviorPolicyDigest)
BEHAVIOR_DIGEST="${BEHAVIOR_DIGEST:-0c4e59831df24e935ad34f65414abb791ddd4dd2d013cf28d0292715cde698d9}"
DEST="${DEST:-$GCS_ROOT/step-check/out}"
WORK="${WORK:-$HOME/r5/step-check}"
CODEROOT="${CODEROOT:-$HOME/code/rl-awire-${SHA:0:7}}"
CODE="$CODEROOT/rl"
PY="${PY:-$HOME/m9-bench/venv/bin/python}"
BRIDGE="${BRIDGE:-$HOME/code/r5s841c38e/rl/integration/bin/model9/Anthropocene.Model9Bridge.dll}"
BRIDGE_SHA256="${BRIDGE_SHA256:-99131dce9c7f0d0d66988b05c3c3e9adea6f6ad3407ce4cd0366d6a14b25eb6b}"
WORLDS="${WORLDS:-91003,91004,91005}"
TURNS="${TURNS:-8}"
# The training worlds (see the header); MOVEMENT=--movement or --no-movement (fixture).
PLAYERS="${PLAYERS:-16}"; WIDTH="${WIDTH:-128}"; HEIGHT="${HEIGHT:-64}"; SEPARATION="${SEPARATION:-9}"
MOVEMENT="${MOVEMENT:---movement}"
case "$MOVEMENT" in --movement) MOVEMENT_ON=True ;; --no-movement) MOVEMENT_ON=False ;; *) echo "bad MOVEMENT $MOVEMENT" >&2; exit 2 ;; esac
case "$SHA$CKPT_SHA256$BEHAVIOR_DIGEST" in *__*) echo "unfilled SHA/CKPT_SHA256/BEHAVIOR_DIGEST" >&2; exit 2 ;; esac
UPLOAD_RESERVE_S="${UPLOAD_RESERVE_S:-900}"
MIN_MEM_GB="${MIN_MEM_GB:-8}"
START_S=$(date +%s)

ts() { date -u +%FT%TZ; }
LAUNCH=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$WORK/logs" "$WORK/out" "$WORK/ckpt"
LOG_FILE="$WORK/logs/job-$LAUNCH.log"
CUR_STAGE=startup
log() { echo "$(ts) [step-check-collect] $*" | tee -a "$LOG_FILE" >&2; }
die() { log "FAIL[$CUR_STAGE] $*"; exit 1; }
sha256_of() { sha256sum "$1" | cut -d' ' -f1; }
gcs_put() { timeout 600 gcloud storage cp --no-user-output-enabled "$1" "$2" < /dev/null || { log "upload failed: $1 -> $2"; return 1; }; }
remote_has() { timeout 120 gcloud storage ls "$1" > /dev/null 2>&1 < /dev/null; }
mem_available_gb() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
TAG="rollout-$(echo "$WORLDS" | tr ',' '-')-t$TURNS-p$PLAYERS"
ROLLOUT="$WORK/out/$TAG.pt"

# Duplicate guards (before the EXIT trap: a refused duplicate must not write EXIT_STATUS).
exec 9> "$WORK/job.lock"
flock -n 9 || { log "another step-check collect_job.sh holds $WORK/job.lock; exiting"; exit 3; }
if pgrep -f "[l]abor_training collect .*$WORK/out/" > /dev/null; then
  log "a step-check collect is already running: $(pgrep -af "[l]abor_training collect .*$WORK/out/" | head -2); exiting"
  exit 3
fi
if remote_has "$DEST/EXIT_STATUS"; then
  log "$DEST/EXIT_STATUS already exists ($(timeout 120 gcloud storage cat "$DEST/EXIT_STATUS" < /dev/null 2> /dev/null | head -1)); nothing to do (use a new DEST to rerun)"
  exit 0
fi

SKIP_STATUS=0   # 1: interrupted (VM stop) -> no EXIT_STATUS, rerun the same line
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
      gcs_put "$WORK/EXIT_STATUS" "$DEST/EXIT_STATUS"   # last and once; this job never stops the VM
    fi
  fi
  exit "$status"
}
trap finish EXIT
# A TERM is either a VM stop or the JOB_TIMEOUT backstop; the collect has its own deadline below,
# so a TERM reaching here is treated as an interruption (rerun the same line).
trap 'SKIP_STATUS=1; log "signal received: interrupted, no EXIT_STATUS (rerun the same line to resume)"; exit 143' TERM INT HUP

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
    if command -v git > /dev/null 2>&1; then  # best-effort local snapshot commit (provenance only)
      (cd "$CODEROOT.tmp/rl" && git init -q && git add -A -f \
        && git -c user.name=step-check -c user.email=step-check@localhost commit -q -m "snapshot of $archived (git archive)") \
        > /dev/null 2>&1 || log "WARN snapshot commit failed"
    fi
    mv "$CODEROOT.tmp" "$CODEROOT"
    touch "$CODEROOT/.snapshot-ok"
  fi
  SOURCE_COMMIT=$(cat "$CODEROOT/COMMIT")
  export PYTHONPATH="$CODE/src" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  export ANTHROPOCENE_DOTNET="${ANTHROPOCENE_DOTNET:-/usr/bin/dotnet}"
  export ANTHROPOCENE_MODEL9_BRIDGE="$BRIDGE"
  [ -f "$BRIDGE" ] || die "missing bridge $BRIDGE (expected the shared r5s841c38e build used by the panel/DAgger jobs; no rebuild is attempted)"
  [ "$(sha256_of "$BRIDGE")" = "$BRIDGE_SHA256" ] || die "bridge sha256 differs from the panel/DAgger bridge (set BRIDGE_SHA256 if a rebuild is intended)"
  [ -x "$ANTHROPOCENE_DOTNET" ] || die "missing dotnet $ANTHROPOCENE_DOTNET"
  local where help_text
  where=$(cd "$CODE" && "$PY" -c 'import anthropocene_rl,sys; print(anthropocene_rl.__file__)' 2>&1) || die "cannot import anthropocene_rl: $where"
  case "$where" in "$CODE"/src/*) ;; *) die "anthropocene_rl resolves to $where, not $CODE/src" ;; esac
  help_text=$(cd "$CODE" && "$PY" -m anthropocene_rl.autonomous_v7.labor_training collect --help < /dev/null 2>&1 || true)
  case "$help_text" in *--allow-vm-scale*) ;; *) die "code $SOURCE_COMMIT lacks labor_training collect --allow-vm-scale" ;; esac
  case "$help_text" in *--movement*--separation*|*--separation*--movement*) ;; *) die "code $SOURCE_COMMIT lacks collect world options (--players/--separation/--movement)" ;; esac
  (cd "$CODE" && "$PY" -c 'from anthropocene_rl.autonomous_v7 import labor_a_ppo' < /dev/null) >> "$LOG_FILE" 2>&1 \
    || die "code $SOURCE_COMMIT lacks labor_a_ppo (WP-A1)"
  log "code $SOURCE_COMMIT in $CODE; bridge sha256 $(sha256_of "$BRIDGE" | cut -c1-16); free mem $(mem_available_gb) GB; free disk $(df -BG --output=avail "$WORK" | tail -1 | tr -d ' ') ; cpus $(getconf _NPROCESSORS_ONLN)"
}

stage_checkpoint() {
  CUR_STAGE=checkpoint
  CKPT="$WORK/ckpt/checkpoint-1.pt"
  if [ ! -f "$CKPT" ] || [ "$(sha256_of "$CKPT")" != "$CKPT_SHA256" ]; then
    rm -f "$CKPT"
    timeout 900 gcloud storage cp --no-user-output-enabled "$CKPT_GCS" "$CKPT" < /dev/null || die "cannot fetch $CKPT_GCS"
  fi
  [ "$(sha256_of "$CKPT")" = "$CKPT_SHA256" ] || die "checkpoint sha256 $(sha256_of "$CKPT") != $CKPT_SHA256"
  # The collect's own loader and world guards, before any native work.
  # The collect's own loader and world guards, plus the step-check identity (one actor-only
  # apply from the D7 start checkpoint on the D7 rollout; critic untouched; the actor whose
  # behaviour digest was recorded), before any native work.
  (cd "$CODE" && "$PY" - "$CKPT" "$WORLDS" "$BEHAVIOR_DIGEST" <<'PY') >> "$LOG_FILE" 2>&1 || die "checkpoint/world preflight failed (see job log)"
import sys, torch
from anthropocene_rl.autonomous_v7 import labor_training as T
from anthropocene_rl.autonomous_v7.labor_model import labor_checkpoint_personality
from anthropocene_rl.autonomous_v7.labor_value_calibration import labor_critic_digest
from anthropocene_rl.autonomous_v7.labor_worlds import require_training_worlds
from anthropocene_rl.autonomous_v7.value_calibration import behavior_policy_digest
state = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
T._require_trainer_checkpoint(state)
_, actor, task, costs, normalizer = T._models_from_checkpoint(state)
worlds = require_training_worlds(T._parse_worlds(sys.argv[2], allow_vm_scale=True), "step-check preflight")
overlap = sorted(set(worlds) & set(state["critic_training_world_seeds"]))
assert not overlap, f"critic-training worlds {overlap}"
assert not set(worlds) & {91000, 91001, 91002}, "D7 worlds reused"
critic = labor_critic_digest(task, costs)
assert critic.startswith("ed0758ed258eb100"), critic
behavior = behavior_policy_digest(actor, T.labor_behavior_configuration(labor_checkpoint_personality(state)))
assert behavior == sys.argv[3], f"checkpoint behaviour digest {behavior} != {sys.argv[3]}"
assert state["iteration"] == 1, state["iteration"]
(check,) = state["actor_step_checks"]
assert check["inputCheckpointSha256"].startswith("14e749b0"), check["inputCheckpointSha256"]
assert check["rolloutSha256"].startswith("d3c75bba"), check["rolloutSha256"]
assert check["criticUpdated"] is False and check["criticOptimizerUpdated"] is False
assert check["outputBehaviorPolicyDigest"] == behavior
print("preflight ok: critic", critic[:16], "normalizer", T._normalization_digest(normalizer)[:16],
      "behaviour", behavior[:16], "iteration", state["iteration"], "code", check["codeCommit"][:12],
      "worlds", worlds)
PY
  log "checkpoint sha256 ${CKPT_SHA256:0:16} ok; preflight: $(tail -1 "$LOG_FILE")"
}

stage_collect() {
  CUR_STAGE=collect
  if remote_has "$DEST/$TAG.pt.sha256"; then
    log "$DEST/$TAG.pt is already uploaded (an earlier launch); not recollecting"
    return 0
  fi
  local budget=$(( $(date +%s) - START_S ))
  case "$JOB_TIMEOUT" in
    *h) budget=$(( ${JOB_TIMEOUT%h} * 3600 - budget - UPLOAD_RESERVE_S )) ;;
    *m) budget=$(( ${JOB_TIMEOUT%m} * 60 - budget - UPLOAD_RESERVE_S )) ;;
    *s) budget=$(( ${JOB_TIMEOUT%s} - budget - UPLOAD_RESERVE_S )) ;;
    *) budget=$(( JOB_TIMEOUT - budget - UPLOAD_RESERVE_S )) ;;
  esac
  [ "$budget" -gt 600 ] || die "no time budget left for the collect ($budget s)"
  local waited=0
  while [ "$(mem_available_gb)" -lt "$MIN_MEM_GB" ]; do
    [ "$waited" -eq 0 ] && log "MemAvailable $(mem_available_gb) GB < $MIN_MEM_GB GB; holding the collect"
    waited=$(( waited + 60 )); sleep 60
    [ "$waited" -lt "$budget" ] || die "memory never freed up"
  done
  rm -f "$ROLLOUT"
  local clog="$WORK/logs/collect-$LAUNCH.log" rc=0 start
  start=$(date +%s)
  log "collect worlds=$WORLDS turns=$TURNS budget=${budget}s (mem $(mem_available_gb) GB) -> $ROLLOUT"
  (cd "$CODE" && nice -n 10 timeout --signal=TERM --kill-after=60 "$budget" "$PY" -m anthropocene_rl.autonomous_v7.labor_training \
      collect --checkpoint "$CKPT" --worlds "$WORLDS" --turns "$TURNS" --allow-vm-scale \
      --players "$PLAYERS" --width "$WIDTH" --height "$HEIGHT" --separation "$SEPARATION" "$MOVEMENT" \
      --out "$ROLLOUT" >> "$clog" 2>&1 < /dev/null) || rc=$?
  local minutes; minutes=$(awk -v s="$(( $(date +%s) - start ))" 'BEGIN{printf "%.1f", s/60}')
  gcs_put "$clog" "$DEST/logs/collect-$LAUNCH.log" || true
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then CUR_STAGE=timeout; die "collect hit its ${budget}s deadline after $minutes min"; fi
  [ "$rc" -eq 0 ] && [ -s "$ROLLOUT" ] || die "collect failed rc=$rc after $minutes min (log $clog)"
  (cd "$CODE" && "$PY" - "$ROLLOUT" "$WORLDS" "$TURNS" "$CKPT_SHA256" "$PLAYERS" "$WIDTH" "$HEIGHT" "$SEPARATION" "$MOVEMENT_ON" "$BEHAVIOR_DIGEST" <<'PY') > "$WORK/out/$TAG.summary.json" 2>> "$LOG_FILE" || die "rollout does not load or differs from the requested collect"
import json, sys
from collections import Counter
from anthropocene_rl.autonomous_v7 import labor_training as T
path, worlds, turns, ckpt, players, width, height, separation, movement, behavior = sys.argv[1:]
batch, _, _, meta = T._load_rollout(path)
batch.require_behaviour_head_logp(); batch.require_cost_multipliers()
rows = meta["decisionRows"]
assert len(rows) == batch.parent_count and all(r.get("population") is not None for r in rows)
expected_worlds = [int(w) for w in worlds.split(",")]
assert meta["worldSeeds"] == expected_worlds, meta["worldSeeds"]
assert int(meta["turns"]) == int(turns), meta["turns"]
assert meta["sourceCheckpointSha256"] == ckpt, meta["sourceCheckpointSha256"]
# on-policy for checkpoint-1: the rollout is bound to the UPDATED actor
assert batch.behavior_policy_digest == behavior == meta["behaviorPolicyDigest"], (batch.behavior_policy_digest, behavior)
expected_reset = {"players": int(players), "movement": movement == "True", "withheld": 0,
                  "width": int(width), "height": int(height), "separation": int(separation)}
assert meta.get("resetOptions") == expected_reset, meta.get("resetOptions")
assert meta.get("resetOptionsSource") == "explicit", meta.get("resetOptionsSource")
per_world = Counter(int(r["world"]) for r in rows)
assert set(per_world) <= set(expected_worlds), per_world
assert all(per_world[w] > 0 for w in expected_worlds), f"a world produced no decisions: {per_world}"
print(json.dumps({"parents": batch.parent_count, "retries": int(len(batch.retry_parent)),
                  "parentsPerWorld": {str(w): per_world[w] for w in expected_worlds},
                  "decisionTurnsPerWorld": {str(w): len({r["turn"] for r in rows if int(r["world"]) == w})
                                            for w in expected_worlds},
                  "worldSeeds": meta["worldSeeds"], "turns": meta["turns"],
                  "resetOptions": meta["resetOptions"],
                  "sourceCheckpointSha256": meta["sourceCheckpointSha256"],
                  "behaviorPolicyDigest": meta["behaviorPolicyDigest"],
                  "terminated": sum(bool(r["terminated"]) for r in rows)}, sort_keys=True))
PY
  sha256_of "$ROLLOUT" > "$WORK/out/$TAG.pt.sha256"
  log "collect done after $minutes min: $(cat "$WORK/out/$TAG.summary.json"); $(du -h "$ROLLOUT" | cut -f1)"
  COLLECT_MINUTES=$minutes
  # Payload first, then its summary, the .sha256 marker last (its presence means complete).
  gcs_put "$ROLLOUT" "$DEST/$TAG.pt" || die "rollout upload failed"
  gcs_put "$WORK/out/$TAG.summary.json" "$DEST/$TAG.summary.json" || die "summary upload failed"
  gcs_put "$WORK/out/$TAG.pt.sha256" "$DEST/$TAG.pt.sha256" || die "sha256 upload failed"
}

stage_final() {
  CUR_STAGE=final
  remote_has "$DEST/$TAG.pt.sha256" || die "rollout marker missing in $DEST"
  cat > "$WORK/config-$LAUNCH.json" <<JSON
{"schema": "r5-step-check-collect-run-v1", "source_commit": "$SOURCE_COMMIT", "checkpoint_sha256": "$CKPT_SHA256",
 "behavior_policy_digest": "$BEHAVIOR_DIGEST",
 "bridge_sha256": "$BRIDGE_SHA256", "worlds": "$WORLDS", "turns": $TURNS, "allow_vm_scale": true,
 "reset": {"players": $PLAYERS, "width": $WIDTH, "height": $HEIGHT, "separation": $SEPARATION, "movement": "$MOVEMENT"},
 "rollout": "$DEST/$TAG.pt", "collect_minutes": "${COLLECT_MINUTES:-reused}", "launch": "$LAUNCH",
 "omp_threads": 1, "nice": 10, "job_timeout": "$JOB_TIMEOUT"}
JSON
  gcs_put "$WORK/config-$LAUNCH.json" "$DEST/config-$LAUNCH.json" || true
  echo "0 $(ts) $TAG source=$SOURCE_COMMIT ckpt=${CKPT_SHA256:0:16} collect_min=${COLLECT_MINUTES:-reused}" > "$WORK/EXIT_STATUS"
  log "done; EXIT_STATUS 0 (uploaded last by the exit handler); the VM is left running"
}

log "start launch=$LAUNCH sha=$SHA worlds=$WORLDS turns=$TURNS reset=$PLAYERS/${WIDTH}x$HEIGHT/sep$SEPARATION/$MOVEMENT dest=$DEST cap=$JOB_TIMEOUT"
rm -f "$WORK/EXIT_STATUS"
stage_code
stage_checkpoint
stage_collect
stage_final
