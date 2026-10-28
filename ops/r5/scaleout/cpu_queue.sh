#!/usr/bin/env bash
# r5 scale-out, CPU VM (anthropocene-trainer-60: 60 vCPU, 118 GB).
#
# One scheduler (cpu_sched.py) runs, under one slot and memory budget:
#   * search with full candidate payloads (labor-advantage-candidates-v1) on H12 roots:
#     reserved worlds 90154-90159 (--held-out-test, 32 roots/world) and fitting worlds
#     90100-90153, 90160-90165 (16 roots/world), H12 teacher roots, 2 re-evaluation streams,
#     horizons 1,3, controls and T2 plans (horizon 4) each on a keyed 25% of roots;
#   * the actor-pilot development panel (24 worlds x 50 decisions), as soon as the GPU VM
#     publishes the pilot actor to $SCALEOUT_GCS/actor-pilot/ (fetched and sha256-checked);
#   * optional actor-visited roots (ACTOR_ROOTS=1; off by default, see README).
# Outputs are uploaded per finished task to $SCALEOUT_GCS/{search,panel}/; milestones
# RESERVED_H12_DONE, PHASE1_DONE (reserved + 8 roots/world), SEARCH_H12_DONE, PANEL_DONE.
# The EXIT trap uploads the scheduler state and shuts the VM down, on success and on
# failure (NO_SHUTDOWN=1 keeps it up). Rerun the same command after any stop: finished
# tasks are skipped, interrupted ones resume (labor_search --resume) or rerun (panel).
#
# Usage on the VM (detached):
#   CODE=$HOME/code/r5s/rl setsid nohup bash $CODE/ops/r5/scaleout/cpu_queue.sh \
#     > $HOME/r5/scaleout/queue.out 2>&1 < /dev/null &
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "$here/lib.sh"
STAGE_NAME=cpu-queue

CODE="${CODE:-$(cd "$here/../../.." && pwd)}"
PY="${PY:-$HOME/m9-bench/venv/bin/python}"
export WORK="${WORK:-$HOME/r5/scaleout}"
export NORMALIZER="${NORMALIZER:-$HOME/r5/actor-normalizer.pt}"
export ANTHROPOCENE_DOTNET="${ANTHROPOCENE_DOTNET:-/usr/bin/dotnet}"
export ANTHROPOCENE_MODEL9_BRIDGE="${ANTHROPOCENE_MODEL9_BRIDGE:-$CODE/integration/bin/model9/Anthropocene.Model9Bridge.dll}"
export PYTHONPATH="$CODE/src" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export CODE PY SCALEOUT_GCS
export ACTOR="${ACTOR:-$WORK/actor/actor-bc-pilot.pt}"
export ACTOR_GCS="${ACTOR_GCS:-$SCALEOUT_GCS/actor-pilot}"
export PANEL_REPORT="${PANEL_REPORT:-$here/panel_report.py}"

mkdir -p "$WORK"
LOG_FILE="$WORK/queue.log"
memwatch_pid=""
exec 8> "$WORK/queue.lock"
flock -n 8 || { echo "another cpu_queue.sh holds $WORK/queue.lock" >&2; exit 3; }

finish() {
  local status=$?
  trap - EXIT
  [ -n "$memwatch_pid" ] && kill "$memwatch_pid" 2> /dev/null || true
  if [ "$status" -eq 0 ]; then log "queue finished"; else log "queue ended status=$status"; fi
  [ -f "$WORK/memwatch.csv" ] && log "memory: $(memwatch_peak "$WORK/memwatch.csv")"
  echo "$status $(ts)" > "$WORK/EXIT_STATUS"
  for f in queue.log queue.out sched.log status.json durations.jsonl memwatch.csv EXIT_STATUS; do
    [ -f "$WORK/$f" ] && gcs_put "$WORK/$f" "$SCALEOUT_GCS/cpu-logs/$f"
  done
  [ -d "$WORK/logs" ] && gcs_sync "$WORK/logs" "$SCALEOUT_GCS/cpu-logs/tasks"
  [ -d "$WORK/FAILED" ] && gcs_sync "$WORK/FAILED" "$SCALEOUT_GCS/cpu-logs/FAILED"
  shutdown_vm
  exit "$status"
}
trap finish EXIT
trap 'log "signal received"; exit 143' TERM INT HUP

# Preflight -----------------------------------------------------------------------------
log "start code=$CODE work=$WORK"
# Persistence left from collection would relaunch the r5 driver or the G0 loop at boot;
# both can stop the VM ("collection complete" / G0 ALLDONE). They must be gone first.
if crontab -l 2> /dev/null | grep -Eq 'g0loop\.sh|r5/loop\.sh'; then
  [ "${ALLOW_CRONTAB:-0}" = "1" ] || die "crontab still has the G0/r5 @reboot lines (crontab -l); remove them (README step C0) or set ALLOW_CRONTAB=1"
fi
if pgrep -f 'r5/(loop|drive)\.sh|g0loop\.sh|g0pipe\.sh' > /dev/null; then
  die "the r5 collection driver or the G0 loop is running (pgrep -af 'loop.sh|drive.sh|g0pipe'); stop it first"
fi
if pgrep -f "labor_search .*--out $WORK/|labor_pretrain_eval autonomous .*--out $WORK/" > /dev/null; then
  die "search/panel processes from an earlier scheduler are still running; kill them (README, failure procedure)"
fi
require_file "$PY" "venv"
require_file "$ANTHROPOCENE_MODEL9_BRIDGE" "bridge of the learner checkout (~/git/checkout.sh builds it)"
require_file "$NORMALIZER" "r5 actor normalizer"
[ "$(sha256_of "$NORMALIZER")" = "$ACTOR_NORMALIZER_SHA256" ] || die "actor normalizer sha256 mismatch"
help_text=$("$PY" -m anthropocene_rl.autonomous_v7.labor_search --help < /dev/null 2>&1 || true)
case "$help_text" in
  *--plan-fraction*) ;;
  *) die "learner checkout lacks labor_search --plan-fraction (needs ops/r5-scaleout eb93ff3 or later)" ;;
esac
[ -z "$(git -C "$CODE" status --porcelain 2> /dev/null)" ] || die "learner checkout is dirty: search refuses non-production identities"
free_gb=$(disk_free_gb "$WORK")
[ "$free_gb" -ge "${MIN_DISK_GB:-40}" ] || die "only ${free_gb} GB free under $WORK"
log "learner $(git -C "$CODE" rev-parse --short HEAD) disk ${free_gb} GB free MemAvailable $(mem_available_gb) GB nproc $(nproc)"
memwatch_pid=$(memwatch_start "$WORK/memwatch.csv" 120)

# Scheduler -------------------------------------------------------------------------------
"$PY" "$here/cpu_sched.py" --work "$WORK" --code "$CODE" --python "$PY" --normalizer "$NORMALIZER" \
  >> "$WORK/sched.out" 2>&1 < /dev/null
