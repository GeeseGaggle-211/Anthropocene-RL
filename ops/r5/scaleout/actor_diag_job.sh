#!/usr/bin/env bash
# r5 actor closed-loop diagnostics (CPU VM): takeover:K and swap:GROUPS on development worlds.
#
# Derived from the actor panel job (actor_panel_job.sh: code/bridge/checkpoint staging, sha
# checks, duplicate guards, memory-gated slot scheduler, per-task resume, timeouts, logging and
# upload helpers). One task = one (world, mode) process:
#   labor_pretrain_eval autonomous --worlds W --modes M --jobs 1
# with the panel's parameters (50 turns, 16 players, 128x64, separation 9, policy seed 0, the
# actor checkpoint as --checkpoint). The evaluator always adds its "heuristic" baseline run, so
# each output holds runs [heuristic, M]. Outputs: $DEST/world-<W>-<M with ':' and '+' -> '_'>.json,
# uploaded once each, raw (no baseline merge, no panel report: analysis happens elsewhere).
# Tasks run mode-major (every world of the first mode first), so partial results are usable.
#
# Modes (see labor_pretrain_eval): takeover:K = the deterministic actor decides decisions
# 0..K-1, h12 every later decision of the same episode; swap:GROUPS = h12 with the '+'-joined
# head groups (labor, specialty, activity, movement, fission, trade) taken from the
# deterministic actor on the same state.
#
# Probe (before the run; SKIP_PROBE=1 skips; remembered per code+bridge+checkpoint): on world
# 2000 with --turns 5, takeover:0 must reproduce the first 5 turn records (all fields) and the
# reset identity of the h12 run in $BASELINE_GCS/world-2000.json, and takeover:99 those of the
# deterministic run in $R1_PANEL_GCS/world-2000.json. A 5-turn episode equals the first 5 turns
# of a 50-turn one: --turns only bounds the decision loop (the reset takes no horizon, and the
# turn records depend only on the steps). Any difference -> FAIL:probe.
#
# GCS status objects. The VM service account cannot overwrite or delete objects, so:
#   * $DEST/EXIT_STATUS is uploaded exactly once, at the end ("0 <utc> ..." or
#     "FAIL:<stage> <utc> ..."). If it already exists at start, the job logs and exits 0 without
#     doing anything: a FAIL finalises this DEST (relaunch with a new DEST).
#   * An interrupted launch (TERM/INT/HUP, e.g. a VM stop) uploads no EXIT_STATUS, so the same
#     line can be rerun: present outputs are skipped, an interrupted task reruns from its reset.
#   * $DEST/RUNNING is the start marker (a failed re-upload on relaunch is harmless and logged).
#   * Objects a relaunch could upload again go to fresh names: logs/job-<launch>.log,
#     logs/<task>-<launch>.log, logs/probe-<launch>.json, config-<launch>.json.
#     CHECKPOINT_SHA256 is uploaded only when absent (a different one refuses this DEST).
#
# Shares the CPU VM: never stops it, nice 10, SLOTS concurrent tasks, a memory floor before each
# launch, no crontab or other persistence.
#
# Required env:  SHA (commit of the code tarball)  CODE_GCS (gs:// path of the code tarball)
#   ACTOR_GCS (gs:// folder of the actor checkpoint)  ACTOR_NAME (checkpoint file name)
#   Use the checkpoint of the R1 panel ($R1_PANEL_GCS): the probe compares deterministic turns.
# Optional env:  MODES="takeover:1,takeover:3,takeover:10,takeover:25"
#   WORLDS="2008 2013 2019 2002 2006 2016" (space list; ranges a-b allowed)  SLOTS=4
#   MIN_MEM_GB=10  TASK_TIMEOUT=12h  LAUNCH_GAP_S=20  WAIT_H=0 (hours to wait for the checkpoint
#   folder's EXIT_STATUS)  ALLOW_FAILED_STATUS=1  ALLOW_NO_STATUS=1  SKIP_PROBE=1
#   BASELINE_GCS=gs://.../r5/scaleout/panel  R1_PANEL_GCS=gs://.../r5/scaleout/panel-r1
#   PY=... BRIDGE=... BRIDGE_SHA256=... WORK=... DEST=gs://.../r5/scaleout/actor-diag
#
# Launch (detached; rerun the same line after an interruption):
#   SHA=<sha> CODE_GCS=<tarball> ACTOR_GCS=<folder> ACTOR_NAME=<file.pt> \
#     setsid nohup bash $HOME/actor_diag_job.sh > $HOME/r5/actor-diag.out 2>&1 < /dev/null &
set -euo pipefail

SHA="${SHA:?set SHA to the code commit of the CODE_GCS tarball}"
CODE_GCS="${CODE_GCS:?set CODE_GCS to the gs:// path of the code tarball (git archive of SHA)}"
ACTOR_GCS="${ACTOR_GCS:?set ACTOR_GCS to the gs:// folder of the actor checkpoint (the R1 panel checkpoint)}"
ACTOR_NAME="${ACTOR_NAME:?set ACTOR_NAME to the checkpoint file name in ACTOR_GCS}"
GCS_ROOT="${GCS_ROOT:-gs://anthropocene-rl-datasets-508912/model9/r5}"
BASELINE_GCS="${BASELINE_GCS:-gs://anthropocene-rl-datasets-508912/model9/r5/scaleout/panel}"
R1_PANEL_GCS="${R1_PANEL_GCS:-gs://anthropocene-rl-datasets-508912/model9/r5/scaleout/panel-r1}"
DEST="${DEST:-$GCS_ROOT/scaleout/actor-diag}"
WORK="${WORK:-$HOME/r5/actor-diag}"
CODEROOT="${CODEROOT:-$HOME/code/rl-diag-$SHA}"
CODE="$CODEROOT/rl"
PY="${PY:-$HOME/m9-bench/venv/bin/python}"
BRIDGE="${BRIDGE:-$HOME/code/r5s841c38e/rl/integration/bin/model9/Anthropocene.Model9Bridge.dll}"
BRIDGE_SHA256="${BRIDGE_SHA256:-99131dce9c7f0d0d66988b05c3c3e9adea6f6ad3407ce4cd0366d6a14b25eb6b}"
MODES="${MODES:-takeover:1,takeover:3,takeover:10,takeover:25}"
WORLDS="${WORLDS:-2008 2013 2019 2002 2006 2016}"
SLOTS="${SLOTS:-4}"
MIN_MEM_GB="${MIN_MEM_GB:-10}"
TASK_TIMEOUT="${TASK_TIMEOUT:-12h}"
LAUNCH_GAP_S="${LAUNCH_GAP_S:-20}"
WAIT_H="${WAIT_H:-0}"
# The pilot panel's parameters (cpu_sched.py defaults).
TURNS=50; PLAYERS=16; WIDTH=128; HEIGHT=64; SEPARATION=9; POLICY_SEED=0
PROBE_WORLD=2000; PROBE_TURNS=5

ts() { date -u +%FT%TZ; }
LAUNCH=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$WORK/logs" "$WORK/raw" "$WORK/probe" "$WORK/helpers" "$WORK/ckpt"
LOG_FILE="$WORK/logs/job-$LAUNCH.log"
CUR_STAGE=startup
log() { echo "$(ts) [actor-diag] $*" | tee -a "$LOG_FILE" >&2; }
die() { log "FAIL[$CUR_STAGE] $*"; exit 1; }
sha256_of() { sha256sum "$1" | cut -d' ' -f1; }
gcs_put() { timeout 600 gcloud storage cp --no-user-output-enabled "$1" "$2" < /dev/null || log "upload failed: $1 -> $2"; }
remote_has() { timeout 120 gcloud storage ls "$1" > /dev/null 2>&1 < /dev/null; }
mem_available_gb() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
task_name() { echo "world-$1-${2//[+:]/_}"; }   # task_name W MODE

# Duplicate guards (before the EXIT trap: a refused duplicate must not write EXIT_STATUS). The
# [x] brackets keep pgrep from matching its own command line; this launcher's argv never
# contains the worker command lines.
exec 9> "$WORK/job.lock"
flock -n 9 || { log "another actor_diag_job.sh holds $WORK/job.lock; exiting"; exit 3; }
if pgrep -f "[l]abor_pretrain_eval autonomous .*$WORK/raw/|[d]iag_tools\.py probe .*$WORK/" > /dev/null; then
  log "actor-diag workers are already running: $(pgrep -af "[l]abor_pretrain_eval autonomous .*$WORK/raw/" | head -3); exiting"
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

# --- helper python (probe) -----------------------------------------------------------------
cat > "$WORK/helpers/diag_tools.py" <<'PY'
"""Diagnostics probe: takeover:0 reproduces h12 and takeover:99 reproduces deterministic."""
import argparse
import json
import sys
import time
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def canon(value):
    return json.dumps(value, sort_keys=True)


def run_of(report, mode, label):
    runs = [r for r in report["runs"] if r["mode"] == mode]
    if len(runs) != 1:
        raise SystemExit(f"{label}: expected one {mode!r} run, found modes {[r['mode'] for r in report['runs']]}")
    return runs[0]


def first_difference(new, old):
    for index, (a, b) in enumerate(zip(new, old)):
        if canon(a) != canon(b):
            fields = sorted(k for k in set(a) | set(b) if canon(a.get(k)) != canon(b.get(k)))
            return {"turn_index": index, "fields": fields}
    if len(new) != len(old):
        return {"turn_index": min(len(new), len(old)), "fields": ["<length>"]}
    return None


def cmd_probe(a):
    from anthropocene_rl.autonomous_v7 import labor_pretrain_eval as E

    expected_reset = json.loads(a.reset_options)
    base, r1 = load(a.baseline), load(a.r1)
    for label, report in (("baseline", base), ("r1", r1)):
        if report["worlds"] != [int(a.world)]:
            raise SystemExit(f"{label}: worlds {report['worlds']} != [{a.world}]")
        if report["reset_options"] != expected_reset:
            raise SystemExit(f"{label}: reset options {report['reset_options']} != {expected_reset}")
        if int(report["policy_seed"]) != int(a.policy_seed):
            raise SystemExit(f"{label}: policy seed {report['policy_seed']} != {a.policy_seed}")
    r1_ckpt = r1.get("checkpoint_sha256")
    if r1_ckpt and r1_ckpt != a.checkpoint_sha256:
        raise SystemExit(f"r1 panel used checkpoint {r1_ckpt}, this job has {a.checkpoint_sha256}")
    references = {
        "takeover:0": ("h12", a.baseline_uri, run_of(base, "h12", "baseline")),
        "takeover:99": ("deterministic", a.r1_uri, run_of(r1, "deterministic", "r1")),
    }
    job = {
        "checkpoint": a.checkpoint, "turns": int(a.turns), "reset_options": expected_reset,
        "policy_seed": int(a.policy_seed), "record_critic_trace": False, "world": int(a.world),
    }
    result = {"world": int(a.world), "turns": int(a.turns), "r1_checkpoint_sha256": r1_ckpt,
              "checks": {}}
    ok = True
    for mode, (ref_mode, uri, ref) in references.items():
        if len(ref["turns"]) < int(a.turns):
            raise SystemExit(f"{uri} {ref_mode} has only {len(ref['turns'])} turns")
        started = time.time()
        new = E._run_job(dict(job, mode=mode))
        old_turns = ref["turns"][: int(a.turns)]
        check = {
            "reference": f"{uri} {ref_mode}",
            "same_reset_identity": new["reset_identity"] == ref["reset_identity"],
            "same_turn_records": canon(new["turns"]) == canon(old_turns),
            "first_difference": first_difference(new["turns"], old_turns),
            "new_reset_identity": new["reset_identity"],
            "reference_reset_identity": ref["reset_identity"],
            "initial_rejections": new.get("initial_rejections"),
            "seconds": round(time.time() - started, 1),
        }
        ok = ok and check["same_reset_identity"] and check["same_turn_records"]
        result["checks"][mode] = check
        print(json.dumps({mode: {k: check[k] for k in ("same_reset_identity", "same_turn_records", "first_difference", "seconds")}}), flush=True)
    result["ok"] = ok
    Path(a.out).write_text(json.dumps(result, indent=1, sort_keys=True))
    return 0 if ok else 3


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("probe")
    for name in ("--baseline", "--baseline-uri", "--r1", "--r1-uri", "--world", "--turns",
                 "--checkpoint", "--checkpoint-sha256", "--reset-options", "--policy-seed", "--out"):
        x.add_argument(name, required=True)
    a = p.parse_args()
    return {"probe": cmd_probe}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
PY

# --- task list (mode-major) ----------------------------------------------------------------
worlds=()
for part in $WORLDS; do
  case "$part" in
    *-*) for ((w = ${part%-*}; w <= ${part#*-}; w++)); do worlds+=("$w"); done ;;
    *) worlds+=("$part") ;;
  esac
done
[ "${#worlds[@]}" -gt 0 ] || die "WORLDS is empty"
for w in "${worlds[@]}"; do
  case "$w" in ''|*[!0-9]*) die "bad world '$w'" ;; esac
  { [ "$w" -ge 2000 ] && [ "$w" -le 2059 ]; } || die "world $w is outside the development range 2000-2059 (3000-3059 are sealed and never evaluated)"
done
IFS=',' read -r -a modes <<< "$MODES"
[ "${#modes[@]}" -gt 0 ] || die "MODES is empty"
for m in "${modes[@]}"; do
  case "$m" in ''|*[!A-Za-z0-9+:]*) die "bad mode '$m' (no spaces; comma-separated)" ;; esac
done
task_worlds=(); task_modes=()
for m in "${modes[@]}"; do
  for w in "${worlds[@]}"; do task_worlds+=("$w"); task_modes+=("$m"); done
done
total=${#task_worlds[@]}

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
      (cd "$CODEROOT.tmp/rl" && git init -q && git add -A -f         && git -c user.name=diag -c user.email=diag@localhost commit -q -m "snapshot of $archived (git archive)")         > /dev/null 2>&1 || log "WARN snapshot commit failed"
    fi
    mv "$CODEROOT.tmp" "$CODEROOT"
    touch "$CODEROOT/.snapshot-ok"
  fi
  SOURCE_COMMIT=$(cat "$CODEROOT/COMMIT")
  export PYTHONPATH="$CODE/src" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  export ANTHROPOCENE_DOTNET="${ANTHROPOCENE_DOTNET:-/usr/bin/dotnet}"
  export ANTHROPOCENE_MODEL9_BRIDGE="$BRIDGE"
  [ -f "$BRIDGE" ] || die "missing bridge $BRIDGE"
  [ "$(sha256_of "$BRIDGE")" = "$BRIDGE_SHA256" ] || die "bridge sha256 differs from the panel bridge (set BRIDGE_SHA256 if the rebuild is intended)"
  [ -x "$ANTHROPOCENE_DOTNET" ] || die "missing dotnet $ANTHROPOCENE_DOTNET"
  local where checked
  where=$(cd "$CODE" && "$PY" -c 'import anthropocene_rl,sys; print(anthropocene_rl.__file__)' 2>&1) || die "cannot import anthropocene_rl: $where"
  case "$where" in "$CODE"/src/*) ;; *) die "anthropocene_rl resolves to $where, not $CODE/src" ;; esac
  # The code must know the diagnostic modes and accept every requested one (strict validation).
  checked=$(cd "$CODE" && "$PY" -c 'import sys
from anthropocene_rl.autonomous_v7.labor_pretrain_eval import validate_autonomous_modes
print(",".join(validate_autonomous_modes(sys.argv[1])))' "$MODES" 2>&1 < /dev/null) \
    || die "code $SOURCE_COMMIT rejects MODES=$MODES: $checked"
  log "code $SOURCE_COMMIT in $CODE; modes $checked; bridge sha256 $(sha256_of "$BRIDGE" | cut -c1-16); free mem $(mem_available_gb) GB; nproc $(nproc)"
}

stage_checkpoint() {
  CUR_STAGE=checkpoint
  local waited=0 status_text first
  while :; do
    status_text=$(timeout 120 gcloud storage cat "$ACTOR_GCS/EXIT_STATUS" < /dev/null 2> /dev/null || true)
    first=${status_text%% *}
    if [ "$first" = "0" ]; then break; fi
    if [ -z "$first" ] && [ "${ALLOW_NO_STATUS:-0}" = "1" ] && remote_has "$ACTOR_GCS/$ACTOR_NAME"; then
      log "WARN no EXIT_STATUS in $ACTOR_GCS; ALLOW_NO_STATUS=1 and the checkpoint exists"; break
    fi
    if [ "${first#FAIL:fidelity}" != "$first" ] && [ "${ALLOW_FAILED_STATUS:-0}" = "1" ]; then
      log "WARN checkpoint job status '$status_text' accepted (ALLOW_FAILED_STATUS=1)"
      break
    fi
    case "$first" in
      FAIL:*) die "the checkpoint job failed ($status_text) in $ACTOR_GCS (ALLOW_FAILED_STATUS=1 accepts FAIL:fidelity)" ;;
      RUNNING|"")
        if [ "$(( waited ))" -ge "$(awk -v h="$WAIT_H" 'BEGIN{printf "%d", h*3600}')" ]; then
          die "the checkpoint is not ready: ${status_text:-no EXIT_STATUS} in $ACTOR_GCS (WAIT_H=$WAIT_H; ALLOW_NO_STATUS=1 accepts a present checkpoint without status)"
        fi
        log "waiting for the checkpoint job (status: ${status_text:-absent}); next look in 10 min"
        sleep 600; waited=$(( waited + 600 )) ;;
      *) die "unrecognised EXIT_STATUS '$status_text' in $ACTOR_GCS" ;;
    esac
  done
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
    >> "$LOG_FILE" 2>&1 || die "the checkpoint does not load with weights_only=True (what the evaluator requires)"
  # One checkpoint per DEST: never mix tasks of different checkpoints. Uploaded only when absent.
  local remote_sha
  remote_sha=$(timeout 120 gcloud storage cat "$DEST/CHECKPOINT_SHA256" < /dev/null 2> /dev/null | cut -d' ' -f1 || true)
  echo "$CKPT_SHA256" > "$WORK/CHECKPOINT_SHA256"
  if [ -n "$remote_sha" ] && [ "$remote_sha" != "$CKPT_SHA256" ]; then
    SKIP_STATUS=1
    die "$DEST already holds tasks of checkpoint $remote_sha, not $CKPT_SHA256; use another DEST"
  fi
  [ -n "$remote_sha" ] || gcs_put "$WORK/CHECKPOINT_SHA256" "$DEST/CHECKPOINT_SHA256"
  echo "RUNNING $(ts) launch=$LAUNCH source=$SOURCE_COMMIT ckpt=${CKPT_SHA256:0:16}" > "$WORK/RUNNING"
  gcs_put "$WORK/RUNNING" "$DEST/RUNNING"   # start marker; fails harmlessly (logged) when a relaunch finds it
  WORK="$WORK" CKPT_SHA256="$CKPT_SHA256" SOURCE_COMMIT="$SOURCE_COMMIT" SLOTS="$SLOTS" LAUNCH="$LAUNCH" \
  WORLDS="${worlds[*]}" MODES="$MODES" DEST="$DEST" ACTOR="$ACTOR_GCS/$ACTOR_NAME" BRIDGE_SHA256="$BRIDGE_SHA256" \
  TURNS="$TURNS" PLAYERS="$PLAYERS" WIDTH="$WIDTH" HEIGHT="$HEIGHT" SEPARATION="$SEPARATION" POLICY_SEED="$POLICY_SEED" \
  "$PY" - <<'PY'
import json, os
E = os.environ
json.dump({
    "schema": "r5-actor-diag-run-v1", "launch": E["LAUNCH"], "checkpoint": E["ACTOR"],
    "checkpoint_sha256": E["CKPT_SHA256"], "source_commit": E["SOURCE_COMMIT"],
    "bridge_sha256": E["BRIDGE_SHA256"], "slots": int(E["SLOTS"]), "dest": E["DEST"],
    "modes": E["MODES"].split(","), "worlds": [int(w) for w in E["WORLDS"].split()],
    "order": "mode-major", "jobs_per_task": 1,
    "panel": {k.lower(): int(E[k]) for k in ("TURNS", "PLAYERS", "WIDTH", "HEIGHT", "SEPARATION", "POLICY_SEED")},
}, open(os.path.join(E["WORK"], "config-" + E["LAUNCH"] + ".json"), "w"), indent=1, sort_keys=True)
PY
  gcs_put "$WORK/config-$LAUNCH.json" "$DEST/config-$LAUNCH.json"
  log "checkpoint $ACTOR_NAME sha256 ${CKPT_SHA256:0:16} (checkpoint job status: ${status_text:-n/a})"
}

stage_probe() {
  CUR_STAGE=probe
  [ "${SKIP_PROBE:-0}" != "1" ] || { log "probe skipped (SKIP_PROBE=1)"; return 0; }
  local key="$SOURCE_COMMIT $BRIDGE_SHA256 $CKPT_SHA256"
  if [ "$(cat "$WORK/PROBE_OK" 2> /dev/null)" = "$key" ]; then log "probe passed earlier for this code, bridge and checkpoint"; return 0; fi
  local base="$WORK/probe/baseline-world-$PROBE_WORLD.json" r1="$WORK/probe/r1-world-$PROBE_WORLD.json"
  local reset out="$WORK/probe/probe-$LAUNCH.json"
  timeout 300 gcloud storage cp --no-user-output-enabled "$BASELINE_GCS/world-$PROBE_WORLD.json" "$base" < /dev/null \
    || die "cannot fetch $BASELINE_GCS/world-$PROBE_WORLD.json (the h12 reference)"
  timeout 300 gcloud storage cp --no-user-output-enabled "$R1_PANEL_GCS/world-$PROBE_WORLD.json" "$r1" < /dev/null \
    || die "cannot fetch $R1_PANEL_GCS/world-$PROBE_WORLD.json (the deterministic reference)"
  reset=$(printf '{"players": %d, "movement": true, "withheld": 0, "width": %d, "height": %d, "separation": %d}' \
    "$PLAYERS" "$WIDTH" "$HEIGHT" "$SEPARATION")
  log "probe: world $PROBE_WORLD, $PROBE_TURNS turns: takeover:0 vs h12 ($BASELINE_GCS) and takeover:99 vs deterministic ($R1_PANEL_GCS)"
  (cd "$CODE" && nice -n 10 timeout 3h "$PY" "$WORK/helpers/diag_tools.py" probe \
      --baseline "$base" --baseline-uri "$BASELINE_GCS/world-$PROBE_WORLD.json" \
      --r1 "$r1" --r1-uri "$R1_PANEL_GCS/world-$PROBE_WORLD.json" \
      --world "$PROBE_WORLD" --turns "$PROBE_TURNS" --checkpoint "$CKPT" --checkpoint-sha256 "$CKPT_SHA256" \
      --reset-options "$reset" --policy-seed "$POLICY_SEED" --out "$out" \
      >> "$WORK/logs/probe-$LAUNCH.log" 2>&1 < /dev/null) || {
    [ -f "$out" ] && gcs_put "$out" "$DEST/logs/probe-$LAUNCH.json"
    gcs_put "$WORK/logs/probe-$LAUNCH.log" "$DEST/logs/probe-$LAUNCH.log"
    die "probe failed: takeover:0/takeover:99 do not reproduce h12/deterministic exactly (see logs/probe-$LAUNCH.log)"
  }
  echo "$key" > "$WORK/PROBE_OK"
  gcs_put "$out" "$DEST/logs/probe-$LAUNCH.json"
  gcs_put "$WORK/logs/probe-$LAUNCH.log" "$DEST/logs/probe-$LAUNCH.log"
  log "probe ok: takeover:0 = h12 and takeover:99 = deterministic on world $PROBE_WORLD ($PROBE_TURNS turns)"
}

run_task() {  # run_task W MODE : one process; uploads its log, then its output (last)
  trap - EXIT TERM INT HUP   # the background copy never runs the launcher's exit handler
  local w=$1 mode=$2 name start rc=0 hours
  name=$(task_name "$w" "$mode")
  local raw="$WORK/raw/$name.json" tlog="$WORK/logs/$name-$LAUNCH.log"
  start=$(date +%s)
  rm -f "$raw" "$raw.tmp"   # never accept a stale report of an earlier (interrupted) attempt
  echo "# $(ts) world $w mode $mode launch $LAUNCH" >> "$tlog"
  (cd "$CODE" && nice -n 10 timeout "$TASK_TIMEOUT" "$PY" -m anthropocene_rl.autonomous_v7.labor_pretrain_eval \
    autonomous --checkpoint "$CKPT" --worlds "$w" --turns "$TURNS" --modes "$mode" \
    --players "$PLAYERS" --width "$WIDTH" --height "$HEIGHT" --separation "$SEPARATION" \
    --policy-seed "$POLICY_SEED" --jobs 1 --out "$raw" >> "$tlog" 2>&1 < /dev/null) || rc=$?
  if [ "$rc" -eq 0 ]; then
    "$PY" -c '
import json, sys
d = json.load(open(sys.argv[1]))
assert d["worlds"] == [int(sys.argv[2])] and d["modes"] == [sys.argv[3]], (d["worlds"], d["modes"])
assert [r["mode"] for r in d["runs"]] == ["heuristic", sys.argv[3]], [r["mode"] for r in d["runs"]]
assert d["turns"] == int(sys.argv[4]) and d["policy_seed"] == int(sys.argv[5])
' "$raw" "$w" "$mode" "$TURNS" "$POLICY_SEED" >> "$tlog" 2>&1 || rc=4
  fi
  hours=$(awk -v s="$(( $(date +%s) - start ))" 'BEGIN{printf "%.2f", s/3600}')
  gcs_put "$tlog" "$DEST/logs/$name-$LAUNCH.log"
  if [ "$rc" -ne 0 ] || [ ! -f "$raw" ]; then
    log "task $name FAILED rc=$rc after ${hours} h (log $tlog)"
    return 1
  fi
  timeout 600 gcloud storage cp --no-user-output-enabled "$raw" "$DEST/$name.json" < /dev/null \
    || { log "task $name upload failed"; return 1; }
  log "task $name done after ${hours} h"
}

stage_run() {
  CUR_STAGE=run
  local pending=() i name running=0 waited_mem
  for ((i = 0; i < total; i++)); do
    name=$(task_name "${task_worlds[$i]}" "${task_modes[$i]}")
    if remote_has "$DEST/$name.json"; then log "skip $name (present in $DEST)"; else pending+=("$i"); fi
  done
  log "$(( total - ${#pending[@]} ))/$total tasks present; running ${#pending[@]} (slots=$SLOTS, mode-major)"
  failures=0
  for i in "${pending[@]}"; do
    while [ "$running" -ge "$SLOTS" ]; do
      wait -n || failures=$(( failures + 1 ))
      running=$(( running - 1 ))
    done
    waited_mem=0
    while [ "$(mem_available_gb)" -lt "$MIN_MEM_GB" ]; do
      [ "$waited_mem" -eq 0 ] && log "MemAvailable $(mem_available_gb) GB < $MIN_MEM_GB GB; holding $(task_name "${task_worlds[$i]}" "${task_modes[$i]}")"
      waited_mem=1; sleep 60
    done
    log "launch $(task_name "${task_worlds[$i]}" "${task_modes[$i]}") (running $running/$SLOTS, mem $(mem_available_gb) GB)"
    run_task "${task_worlds[$i]}" "${task_modes[$i]}" &
    running=$(( running + 1 ))
    sleep "$LAUNCH_GAP_S"
  done
  while [ "$running" -gt 0 ]; do
    wait -n || failures=$(( failures + 1 ))
    running=$(( running - 1 ))
  done
}

stage_final() {
  CUR_STAGE=run
  local present=0 i
  for ((i = 0; i < total; i++)); do
    remote_has "$DEST/$(task_name "${task_worlds[$i]}" "${task_modes[$i]}").json" && present=$(( present + 1 ))
  done
  if [ "$present" -ne "$total" ]; then
    echo "FAIL:run $(ts) tasks=$present/$total failed_this_launch=$failures source=$SOURCE_COMMIT ckpt=${CKPT_SHA256:0:16}" > "$WORK/EXIT_STATUS"
    log "$present/$total tasks present, $failures failed this launch; EXIT_STATUS FAIL:run finalises $DEST"
    return 1
  fi
  echo "0 $(ts) tasks=$total/$total source=$SOURCE_COMMIT ckpt=${CKPT_SHA256:0:16} modes=$MODES" > "$WORK/EXIT_STATUS"
  log "all $total tasks present; EXIT_STATUS 0 (uploaded last by the exit handler)"
}

# --- main ----------------------------------------------------------------------------------
failures=0
log "start launch=$LAUNCH sha=$SHA modes=$MODES worlds=${worlds[*]} ($total tasks) slots=$SLOTS dest=$DEST"
rm -f "$WORK/EXIT_STATUS"
stage_code
stage_checkpoint
stage_probe
stage_run
stage_final
