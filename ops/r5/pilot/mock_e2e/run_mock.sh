#!/usr/bin/env bash
# Local end-to-end validation of ppo_pilot_job.sh without the native bridge or the cloud
# (Git Bash on the workstation, or any Linux). Mocks: gcloud (local write-once bucket),
# flock/pgrep (Git Bash lacks them), and the two native stages (driver.py). Train,
# calibration and every pilot check run for real on synthetic fixture files.
#
# Scenarios: (A) ITERATIONS=2 with the iteration-1 actor-gate stage failing -> FAIL-<launch>,
# no EXIT_STATUS, iteration 0 and the iteration-1 collect/train-anchor markers uploaded;
# (B) the local work directory wiped (a fresh VM disk) and the same line rerun -> iteration 0
# and the finished iteration-1 stages restored from the mock bucket, the rest run ->
# EXIT_STATUS; (C) a rerun -> "already complete". Then it checks the uploaded layout.
#
# Usage: MOCK_VENV_PY=<python with torch> bash run_mock.sh <commit-sha> <scratch-dir>
set -euo pipefail
SHA=${1:?commit sha}; SCRATCH=${2:?scratch dir}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
export MOCK_DIR="$HERE" MOCK_REPO="$REPO" MOCK_GCS="$SCRATCH/gcs"
: "${MOCK_VENV_PY:?set MOCK_VENV_PY}"
rm -rf "$SCRATCH"; mkdir -p "$SCRATCH"
if command -v cygpath > /dev/null 2>&1; then  # Git Bash: one canonical spelling (/tmp mount)
  SCRATCH=$(cygpath -u "$(cygpath -w "$SCRATCH")"); MOCK_GCS="$SCRATCH/gcs"
fi
mkdir -p "$MOCK_GCS" "$SCRATCH/home"
export PATH="$HERE/shims:$PATH"

# code tarball, exactly as the VM gets it
GCS_ROOT=gs://mock-bucket/model9/r5
mkdir -p "$MOCK_GCS/mock-bucket/model9/r5/code"
git -C "$REPO" archive --format=tar.gz -o "$MOCK_GCS/mock-bucket/model9/r5/code/rl-pilot-${SHA:0:7}.tar.gz" "$SHA"

# synthetic start checkpoint (the WP-A1 fixture models; iteration 0, empty history)
mkdir -p "$MOCK_GCS/mock-bucket/model9/r5/d7-run"
(cd "$REPO" && PYTHONPATH="src;tests" "$MOCK_VENV_PY" - "$(cygpath -w "$MOCK_GCS/mock-bucket/model9/r5/d7-run/checkpoint-start.pt" 2>/dev/null || echo "$MOCK_GCS/mock-bucket/model9/r5/d7-run/checkpoint-start.pt")" <<'PY'
import sys, torch
from test_labor_a_ppo_wiring import _default_models, _trainer_state
actor, task, costs, normalizer, layout = _default_models()
torch.save(_trainer_state(actor, task, costs, normalizer, layout), sys.argv[1])
PY
)
CKPT_SHA=$(sha256sum "$MOCK_GCS/mock-bucket/model9/r5/d7-run/checkpoint-start.pt" | cut -d' ' -f1)
echo "dummy bridge" > "$SCRATCH/bridge.dll"

run_job() {
  env HOME="$SCRATCH/home" SHA="$SHA" GCS_ROOT="$GCS_ROOT" CKPT_SHA256="$CKPT_SHA" \
    PY="$HERE/shims/mockpy" BRIDGE="$SCRATCH/bridge.dll" \
    BRIDGE_SHA256="$(sha256sum "$SCRATCH/bridge.dll" | cut -d' ' -f1)" \
    ANTHROPOCENE_DOTNET=/usr/bin/true MIN_MEM_GB=0 TURNS=2 ITERATIONS=2 ANCHOR_PATHS=3 \
    JOB_TIMEOUT=30m "$@" bash "$HERE/../ppo_pilot_job.sh"
}
DEST="$MOCK_GCS/mock-bucket/model9/r5/ppo-pilot/pilot-${SHA:0:7}"
WORK="$SCRATCH/home/r5/ppo-pilot/pilot-${SHA:0:7}"

echo "=== A: iteration-1 actor-gate fails"
if run_job MOCK_FAIL_WORLDS=91210,91211 > "$SCRATCH/A.out" 2>&1; then echo "A should fail"; exit 1; fi
ls "$DEST"/FAIL-* > /dev/null
[ ! -e "$DEST/EXIT_STATUS" ]
for f in iter-000/iteration.done.json iter-001/collect.done.json iter-001/train-anchor.done.json; do
  [ -e "$DEST/$f" ] || { echo "A: missing $f"; exit 1; }
done
[ ! -e "$DEST/iter-001/actor-gate.done.json" ]
echo "A ok: $(cat "$DEST"/FAIL-*)"

echo "=== B: fresh disk, same line"
rm -rf "$SCRATCH/home"; mkdir -p "$SCRATCH/home"
run_job > "$SCRATCH/B.out" 2>&1 || { tail -30 "$SCRATCH/B.out"; exit 1; }
grep -q "iter 0: complete" "$SCRATCH/B.out"
grep -q "iter 1 collect: restored" "$SCRATCH/B.out"
grep -q "iter 1 train-anchor: restored" "$SCRATCH/B.out"
grep -q "iter 1 actor-gate: start" "$SCRATCH/B.out"
[ -e "$DEST/EXIT_STATUS" ]
echo "B ok: $(cat "$DEST/EXIT_STATUS")"

echo "=== C: rerun after completion"
run_job > "$SCRATCH/C.out" 2>&1
grep -q "the pilot is complete" "$SCRATCH/C.out"
echo "C ok"

echo "=== layout and summaries"
(cd "$REPO" && PYTHONPATH=src "$MOCK_VENV_PY" - "$(cygpath -w "$WORK" 2>/dev/null || echo "$WORK")" <<'PY'
import json, sys
from pathlib import Path
work = Path(sys.argv[1])
for k in (0, 1):
    s = json.loads((work / f"iter-{k:03d}" / "summary.json").read_text())
    print(k, {"mode": s["actorGateMode"], "gateOpen": (s["gate"] or {}).get("actorGateOpen"),
              "failing": (s["gate"] or {}).get("failingHeads"),
              "accepted": s["actor"]["acceptedSteps"], "attempted": s["actor"]["attemptedSteps"],
              "epochs": len(s["actor"]["perEpoch"]), "criticLoss": round(s["critic"]["loss"], 4),
              "dlogpJoint": s["meanAbsDeltaLogp"]["initialJoint"],
              "bypassRecords": s["pilotBypassUpdatesInCheckpoint"],
              "anchors": s["anchors"], "stages": sorted(s["stageSeconds"])})
    assert s["checkpointIteration"] == {"in": k, "out": k + 1}
    assert s["pilotBypassUpdatesInCheckpoint"] == k + 1
    assert s["anchors"]["trainAnchor"]["rows"] > 0 and s["anchors"]["gateAnchor"]["rankRows"] > 0
PY
)
find "$DEST" -type f | sed "s|$DEST/||" | sort
echo "MOCK E2E PASSED"
