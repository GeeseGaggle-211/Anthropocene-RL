#!/usr/bin/env bash
# WP-B branch cost probe (CPU VM; no GCS, no markers, never stops the VM): one world, TURNS
# decisions, three labor_training collect runs IN PARALLEL from one checkpoint, one thread
# each:
#   off       the plain rollout (reference rollout sha256 and rollout wall time)
#   uniform   --branch-select uniform  --branch-rate RATE                --branch-horizon HORIZON
#   adaptive  --branch-select adaptive --branch-rate RATE --branch-floor FLOOR (H12 shadow on)
# then labor_branch_report on both branch files, and a summary (stdout and OUT/summary.json):
# wall seconds and peak RSS per run, native steps per stage, b seconds per branched state,
# shared-a seconds per branched decision, shadow seconds per decision, achieved selection
# rate, and whether all three rollout.pt files are byte-identical (they must be: branching
# and the shadow never change the rollout; a difference means native fork/restore/capture
# is not pure, and the branch data must not be used).
#
# Required env: SHA (code commit; its tree extracted at CODEROOT/rl, e.g. by
# ppo_pilot_job.sh's code stage, or give TARBALL=<local rl-pilot-<sha7>.tar.gz>), CKPT (an
# input checkpoint path, e.g. a pilot iteration's checkpoint).
# Optional: WORLD (91500; never 3000-3059 or 90154-90159) TURNS (16) RATE (0.1) FLOOR (0.025)
# HORIZON (3) CONTINUATION (actor) PLAYERS WIDTH HEIGHT SEPARATION (16/128/64/9) OUT PY
# BRIDGE BRIDGE_SHA256 CODEROOT.
#
# Launch (detached):
#   SHA=<sha> CKPT=<ckpt> setsid nohup bash $HOME/branch_cost_probe.sh \
#     > $HOME/r5/branch-cost.out 2>&1 < /dev/null &
set -euo pipefail

SHA="${SHA:?export SHA=<code commit>}"
CKPT="${CKPT:?export CKPT=<input checkpoint path>}"
WORLD="${WORLD:-91500}"; TURNS="${TURNS:-16}"
RATE="${RATE:-0.1}"; FLOOR="${FLOOR:-0.025}"; HORIZON="${HORIZON:-3}"
CONTINUATION="${CONTINUATION:-actor}"
PLAYERS="${PLAYERS:-16}"; WIDTH="${WIDTH:-128}"; HEIGHT="${HEIGHT:-64}"; SEPARATION="${SEPARATION:-9}"
CODEROOT="${CODEROOT:-$HOME/code/rl-pilot-${SHA:0:7}}"
CODE="$CODEROOT/rl"
PY="${PY:-$HOME/m9-bench/venv/bin/python}"
BRIDGE="${BRIDGE:-$HOME/code/r5s841c38e/rl/integration/bin/model9/Anthropocene.Model9Bridge.dll}"
BRIDGE_SHA256="${BRIDGE_SHA256:-99131dce9c7f0d0d66988b05c3c3e9adea6f6ad3407ce4cd0366d6a14b25eb6b}"
OUT="${OUT:-$HOME/r5/branch-cost/${SHA:0:7}-w$WORLD-t$TURNS-r$RATE}"

case "$WORLD" in ''|*[!0-9]*) echo "WORLD must be a non-negative integer" >&2; exit 2 ;; esac
if { [ "$WORLD" -ge 3000 ] && [ "$WORLD" -le 3059 ]; } || { [ "$WORLD" -ge 90154 ] && [ "$WORLD" -le 90159 ]; }; then
  echo "world $WORLD is sealed/reserved" >&2; exit 2
fi
[ -f "$CKPT" ] || { echo "missing checkpoint $CKPT" >&2; exit 2; }
if [ -n "${TARBALL:-}" ] && [ ! -f "$CODEROOT/.snapshot-ok" ]; then
  rm -rf "$CODEROOT.tmp"; mkdir -p "$CODEROOT.tmp/rl"
  tar -xzf "$TARBALL" -C "$CODEROOT.tmp/rl"
  rm -rf "$CODEROOT"; mv "$CODEROOT.tmp" "$CODEROOT"; touch "$CODEROOT/.snapshot-ok"
fi
[ -d "$CODE/src" ] || { echo "no code at $CODE (extract the $SHA tarball or set TARBALL)" >&2; exit 2; }
[ -f "$BRIDGE" ] && [ "$(sha256sum "$BRIDGE" | cut -d' ' -f1)" = "$BRIDGE_SHA256" ] \
  || { echo "bridge $BRIDGE missing or sha256 differs" >&2; exit 2; }
export PYTHONPATH="$CODE/src" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export ANTHROPOCENE_DOTNET="${ANTHROPOCENE_DOTNET:-/usr/bin/dotnet}" ANTHROPOCENE_MODEL9_BRIDGE="$BRIDGE"
case "$(cd "$CODE" && "$PY" -m anthropocene_rl.autonomous_v7.labor_training collect --help < /dev/null 2>&1 || true)" in
  *--branch-select*) ;; *) echo "code at $CODE lacks collect --branch-select" >&2; exit 2 ;; esac
mkdir -p "$OUT"
TIMER=""; [ -x /usr/bin/time ] && TIMER="/usr/bin/time -v"
world_flags=(--worlds "$WORLD" --turns "$TURNS" --allow-vm-scale --players "$PLAYERS" --width "$WIDTH"
             --height "$HEIGHT" --separation "$SEPARATION" --movement)
declare -A extra=(
  [off]=""
  [uniform]="--branch-select uniform --branch-rate $RATE --branch-horizon $HORIZON --branch-continuation $CONTINUATION"
  [adaptive]="--branch-select adaptive --branch-rate $RATE --branch-floor $FLOOR --branch-horizon $HORIZON --branch-continuation $CONTINUATION"
)
echo "$(date -u +%FT%TZ) probe sha=$SHA world=$WORLD turns=$TURNS rate=$RATE floor=$FLOOR horizon=$HORIZON continuation=$CONTINUATION out=$OUT"
pids=()
for run in off uniform adaptive; do
  mkdir -p "$OUT/$run"
  # shellcheck disable=SC2086
  ( start=$(date +%s.%N); rc=0
    cd "$CODE" && nice -n 10 $TIMER "$PY" -m anthropocene_rl.autonomous_v7.labor_training collect \
      --checkpoint "$CKPT" "${world_flags[@]}" ${extra[$run]} --out "$OUT/$run/rollout.pt" \
      > "$OUT/$run/collect.log" 2>&1 < /dev/null || rc=$?
    echo "{\"rc\": $rc, \"seconds\": $(awk -v a="$start" -v b="$(date +%s.%N)" 'BEGIN{print b - a}')}" \
      > "$OUT/$run/wall.json"
    exit $rc ) &
  pids+=($!)
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
for run in uniform adaptive; do
  [ -f "$OUT/$run/rollout.branches.pt" ] && (cd "$CODE" && "$PY" -m anthropocene_rl.autonomous_v7.labor_branch_report \
    --branches "$OUT/$run/rollout.branches.pt" --rollout "$OUT/$run/rollout.pt" --out "$OUT/$run/branch-report.json") || failed=1
done
(cd "$CODE" && "$PY" - "$OUT" <<'PY') | tee "$OUT/summary.json"
import hashlib, json, os, re, sys
out = sys.argv[1]
summary = {"runs": {}}
shas = {}
for run in ("off", "uniform", "adaptive"):
    d = os.path.join(out, run)
    entry = json.load(open(os.path.join(d, "wall.json"))) if os.path.exists(os.path.join(d, "wall.json")) else {}
    log = open(os.path.join(d, "collect.log"), errors="replace").read() if os.path.exists(os.path.join(d, "collect.log")) else ""
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", log)
    entry["peakRssGb"] = round(int(m.group(1)) / 1048576, 2) if m else None
    path = os.path.join(d, "rollout.pt")
    if os.path.exists(path):
        shas[run] = hashlib.sha256(open(path, "rb").read()).hexdigest()
        entry["rolloutSha256"] = shas[run]
    rep = os.path.join(d, "branch-report.json")
    if os.path.exists(rep):
        r = json.load(open(rep))
        entry["branches"] = {
            "states": r["states"], "branchedStates": r["branchedStates"],
            "selectionRate": r["selectionRate"], "meanSelectionProbability": r["meanSelectionProbability"],
            "aFirstMatchesRollout": r["aFirstMatchesRollout"], "unavailableTargets": r["unavailableTargets"],
            "ess": r["importanceWeights"]["effectiveSampleSize"], "cost": r["cost"],
            "branchFileGb": round(os.path.getsize(os.path.join(d, "rollout.branches.pt")) / 2**30, 3),
            "meanAbsDiffFolded": r["targets"]["folded"]["meanAbsDiff"],
        }
    summary["runs"][run] = entry
summary["rolloutsIdentical"] = len(shas) == 3 and len(set(shas.values())) == 1
print(json.dumps(summary, indent=1, sort_keys=True))
PY
echo "$(date -u +%FT%TZ) probe done failed=$failed identical=$(grep -o '"rolloutsIdentical": [a-z]*' "$OUT/summary.json")"
exit "$failed"
