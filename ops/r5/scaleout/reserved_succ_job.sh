#!/usr/bin/env bash
# Rerun of the reserved-world H12 search (reserved-h12) with successor critic-input rows
# (labor_search --successor-rows, candidate payload export v2), for the per-(state, action)
# advantage check (labor_sa_advantage successor_check). Self-contained: shares the CPU VM
# with a running cpu_queue.sh / fit-h12 search, never stops the VM, never touches
# $HOME/r5/scaleout.
#
# Same roots as reserved-h12: worlds 90154-90159 x shards t00-13 / t13-25 / t25-38 /
# t38-50 (24 tasks, 8 roots each), the same search arguments, normalizer and bridge as
# cpu_sched.py's reserved-h12 stage (bridge sha256 99131dce..., normalizer 3c8c5ca3...),
# so every root coincides and the v1 fields of each payload equal the original's.
#
#   * code: gs://.../r5/code/rl-succ-<sha>.tar.gz (git archive), unpacked into its own dir
#     and committed there as a local snapshot (labor_search requires a clean learner
#     commit); the tarball's own commit id is recorded in SOURCE_COMMIT and uploaded;
#   * one process per (shard, world) task, SLOTS at a time (default 8), nice 10, timeout;
#   * a task is skipped when its search_report.json is already in $DEST; a finished task
#     uploads its output (search_report.json last, so its presence means complete);
#   * EXIT_STATUS is written and uploaded once all 24 tasks are present in $DEST.
#
# Usage on the VM (detached):
#   SHA=<code sha> setsid nohup bash reserved_succ_job.sh > $HOME/r5/reserved-succ.out 2>&1 < /dev/null &
# Rerun the same command after a stop: finished tasks are skipped, partial ones resume
# (labor_search --resume).
set -euo pipefail

SHA="${SHA:?set SHA to the code commit of gs://.../r5/code/rl-succ-<sha>.tar.gz}"
GCS_ROOT="${GCS_ROOT:-gs://anthropocene-rl-datasets-508912/model9/r5}"
CODE_GCS="${CODE_GCS:-$GCS_ROOT/code/rl-succ-$SHA.tar.gz}"
DEST="${DEST:-$GCS_ROOT/scaleout/search/reserved-h12-succ}"
WORK="${WORK:-$HOME/r5/reserved-succ}"
CODE="${CODE:-$HOME/code/rl-succ-$SHA}"
PY="${PY:-$HOME/m9-bench/venv/bin/python}"
NORMALIZER="${NORMALIZER:-$HOME/r5/actor-normalizer.pt}"
NORMALIZER_SHA256="3c8c5ca300bab11d9651dfc4bfbaa36c7bae08e4c0236773c84e2237f1fa57de"
BRIDGE="${BRIDGE:-$HOME/code/r5s841c38e/rl/integration/bin/model9/Anthropocene.Model9Bridge.dll}"
BRIDGE_SHA256="99131dce9c7f0d0d66988b05c3c3e9adea6f6ad3407ce4cd0366d6a14b25eb6b"
SLOTS="${SLOTS:-8}"
TASK_TIMEOUT="${TASK_TIMEOUT:-20h}"  # historically ~7.3 h per task at 16 slots
WORLDS="${WORLDS:-90154 90155 90156 90157 90158 90159}"
SPLITS="${SPLITS:-0:13 13:25 25:38 38:50}"  # first:turns, 8 roots each (cpu_sched RESERVED_SPLITS)
ROOTS="${ROOTS:-8}"

mkdir -p "$WORK/logs"
LOG="$WORK/job.log"
ts() { date -u +%FT%TZ; }
log() { echo "$(ts) [reserved-succ] $*" | tee -a "$LOG" >&2; }
die() { log "FAIL $*"; exit 1; }
sha256_of() { sha256sum "$1" | cut -d' ' -f1; }

# Duplicate guards: one launcher (flock), and no search workers of an earlier launcher.
# The bracket keeps the pattern from matching pgrep's own command line; the launcher's
# command line never contains the worker output path.
exec 9> "$WORK/job.lock"
flock -n 9 || die "another reserved_succ_job.sh holds $WORK/job.lock"
if pgrep -f '[r]eserved-h12-succ/t' > /dev/null; then
  die "reserved-h12-succ search workers are already running: $(pgrep -af '[r]eserved-h12-succ/t' | head -3)"
fi

# Code ------------------------------------------------------------------------------------
if [ ! -f "$CODE/.snapshot-ok" ]; then
  log "fetching $CODE_GCS"
  rm -rf "$CODE" "$CODE.tmp"
  mkdir -p "$CODE.tmp"
  timeout 600 gcloud storage cp --no-user-output-enabled "$CODE_GCS" "$CODE.tmp/code.tar.gz" < /dev/null \
    || die "download failed: $CODE_GCS"
  # get-tar-commit-id stops reading after the header, so gzip may die of SIGPIPE (141)
  # under pipefail; judge by the output instead of the pipeline status.
  archived=$(gzip -dc "$CODE.tmp/code.tar.gz" | git get-tar-commit-id || true)
  [ -n "$archived" ] || die "tarball has no commit id"
  [ "$archived" = "$SHA" ] || [ "${archived:0:${#SHA}}" = "$SHA" ] || die "tarball commit $archived != SHA $SHA"
  mkdir -p "$CODE.tmp/rl"
  tar -xzf "$CODE.tmp/code.tar.gz" -C "$CODE.tmp/rl"
  (
    cd "$CODE.tmp/rl"
    git init -q
    git add -A -f
    git -c user.name=reserved-succ -c user.email=reserved-succ@localhost \
      commit -q -m "snapshot of $archived (git archive)"
  ) || die "snapshot commit failed"
  echo "$archived" > "$CODE.tmp/SOURCE_COMMIT"
  mv "$CODE.tmp" "$CODE"
  touch "$CODE/.snapshot-ok"
fi
RL="$CODE/rl"
SOURCE_COMMIT=$(cat "$CODE/SOURCE_COMMIT")
SNAPSHOT=$(git -C "$RL" rev-parse HEAD)
[ -z "$(git -C "$RL" status --porcelain --untracked-files=no)" ] || die "code snapshot $RL is dirty"

export PYTHONPATH="$RL/src" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export ANTHROPOCENE_DOTNET="${ANTHROPOCENE_DOTNET:-/usr/bin/dotnet}"
export ANTHROPOCENE_MODEL9_BRIDGE="$BRIDGE"

# Preflight -------------------------------------------------------------------------------
[ -x "$PY" ] || die "missing venv python $PY"
[ -f "$BRIDGE" ] || die "missing bridge $BRIDGE"
[ "$(sha256_of "$BRIDGE")" = "$BRIDGE_SHA256" ] || die "bridge sha256 differs from reserved-h12's"
[ -f "$NORMALIZER" ] || die "missing normalizer $NORMALIZER"
[ "$(sha256_of "$NORMALIZER")" = "$NORMALIZER_SHA256" ] || die "normalizer sha256 mismatch"
help_text=$("$PY" -m anthropocene_rl.autonomous_v7.labor_search --help < /dev/null 2>&1 || true)
case "$help_text" in
  *--successor-rows*) ;;
  *) die "code $SOURCE_COMMIT lacks labor_search --successor-rows" ;;
esac
log "start source=$SOURCE_COMMIT snapshot=${SNAPSHOT:0:12} slots=$SLOTS dest=$DEST nproc=$(nproc)"
echo "$SOURCE_COMMIT" > "$WORK/SOURCE_COMMIT"
timeout 600 gcloud storage cp --no-user-output-enabled "$WORK/SOURCE_COMMIT" "$DEST/SOURCE_COMMIT" < /dev/null \
  || log "upload failed: SOURCE_COMMIT"

remote_done() {  # remote_done REL -> 0 when DEST/REL/search_report.json exists
  timeout 300 gcloud storage ls "$DEST/$1/search_report.json" > /dev/null 2>&1 < /dev/null
}

run_task() {  # run_task FIRST TURNS WORLD (one process; uploads on success)
  local first=$1 turns=$2 world=$3
  local rel; rel=$(printf "t%02d-%02d/w%s" "$first" "$turns" "$world")
  local out="$WORK/reserved-h12-succ/$rel"
  local tlog; tlog="$WORK/logs/$(echo "$rel" | tr '/' '-').log"
  mkdir -p "$(dirname "$out")"
  local start; start=$(date +%s)
  local rc=0
  # cpu_sched.py reserved-h12 command (actor-free H12 roots) plus --successor-rows.
  nice -n 10 timeout "$TASK_TIMEOUT" "$PY" -m anthropocene_rl.autonomous_v7.labor_search \
    --normalizer-checkpoint "$NORMALIZER" \
    --worlds "$world" --source h12 \
    --turns "$turns" --first-turn "$first" \
    --roots-per-world "$ROOTS" --root-key 0 \
    --candidates 14 --horizons 1,3 --keep all,3 --reeval-streams 2 \
    --control-fraction 0.25 --plan-candidates 2 --plan-steps 3 --plan-fraction 0.25 \
    --personality dirichlet:1.0 --personality-key 4242 \
    --players 16 --width 128 --height 64 --separation 9 \
    --jobs 1 --resume --held-out-test --successor-rows \
    --out "$out" > "$tlog" 2>&1 < /dev/null || rc=$?
  local hours; hours=$(awk -v s="$(( $(date +%s) - start ))" 'BEGIN{printf "%.2f", s/3600}')
  if [ "$rc" -ne 0 ] || [ ! -f "$out/search_report.json" ]; then
    log "task $rel failed rc=$rc after ${hours} h (log $tlog)"
    timeout 600 gcloud storage cp --no-user-output-enabled "$tlog" "$DEST/logs/" < /dev/null || true
    return 1
  fi
  # Everything but the completion marker first, then the marker.
  timeout 3600 gcloud storage rsync -r --no-user-output-enabled -x '(^|.*/)search_report\.json$|^_work/.*' \
      "$out" "$DEST/$rel" < /dev/null \
    && timeout 600 gcloud storage cp --no-user-output-enabled "$out/search_report.json" \
      "$DEST/$rel/search_report.json" < /dev/null \
    || { log "upload failed for $rel"; return 1; }
  timeout 600 gcloud storage cp --no-user-output-enabled "$tlog" "$DEST/logs/" < /dev/null || true
  log "task $rel done after ${hours} h, payload $(du -sh "$out" | cut -f1)"
}

# Tasks -----------------------------------------------------------------------------------
tasks=()
for split in $SPLITS; do
  for world in $WORLDS; do
    tasks+=("${split%%:*} ${split##*:} $world")
  done
done
total=${#tasks[@]}
pending=()
for t in "${tasks[@]}"; do
  read -r first turns world <<< "$t"
  rel=$(printf "t%02d-%02d/w%s" "$first" "$turns" "$world")
  if remote_done "$rel"; then log "skip $rel (present in $DEST)"; else pending+=("$t"); fi
done
log "$(( total - ${#pending[@]} ))/$total tasks present; running ${#pending[@]}"

failures=0
running=0
for t in "${pending[@]}"; do
  while [ "$running" -ge "$SLOTS" ]; do
    wait -n || failures=$(( failures + 1 ))
    running=$(( running - 1 ))
  done
  read -r first turns world <<< "$t"
  log "launch t$first-$turns w$world"
  run_task "$first" "$turns" "$world" &
  running=$(( running + 1 ))
  sleep "${LAUNCH_GAP_S:-30}"
done
while [ "$running" -gt 0 ]; do
  wait -n || failures=$(( failures + 1 ))
  running=$(( running - 1 ))
done

present=0
for t in "${tasks[@]}"; do
  read -r first turns world <<< "$t"
  remote_done "$(printf "t%02d-%02d/w%s" "$first" "$turns" "$world")" && present=$(( present + 1 ))
done
timeout 600 gcloud storage cp --no-user-output-enabled "$LOG" "$DEST/logs/job.log" < /dev/null || true
if [ "$present" -eq "$total" ]; then
  echo "0 $(ts) $present/$total source=$SOURCE_COMMIT" > "$WORK/EXIT_STATUS"
  timeout 600 gcloud storage cp --no-user-output-enabled "$WORK/EXIT_STATUS" "$DEST/EXIT_STATUS" < /dev/null \
    || die "EXIT_STATUS upload failed"
  log "all $total tasks present; EXIT_STATUS uploaded"
  exit 0
fi
log "$present/$total tasks present, $failures failed this run; rerun the same command"
exit 1
