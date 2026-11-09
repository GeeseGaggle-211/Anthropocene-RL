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
# (D) exploration knobs: labor_head_temper tempers the start checkpoint against the uploaded
# iteration-0 rollout, the tempered file is uploaded separately, and a 2-iteration run starts
# from it (START_CKPT_GCS + CKPT_SHA256) with HEAD_ENTROPY_FLOOR -> its own PILOT_ID
# (pilot-<sha7>-s<start8>-f<floor8>), train records the floor and per-head entropies, the
# summaries carry headEntropy in/out; a bad floor spec is refused before any stage.
#
# MOCK_TURNS (default 2) is the job's TURNS in every scenario (MOCK_TURNS=16: the next run's
# rollout length; collect and check-rollout get --turns 16, and D leaves JOB_TIMEOUT unset so
# the TURNS-scaled default cap, 14h at 16, is exercised and checked in the start log).
#
# Usage: MOCK_VENV_PY=<python with torch> [MOCK_TURNS=16] bash run_mock.sh <commit-sha> <scratch-dir>
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
    ANTHROPOCENE_DOTNET=/usr/bin/true MIN_MEM_GB=0 TURNS="${MOCK_TURNS:-2}" ITERATIONS=2 ANCHOR_PATHS=3 \
    JOB_TIMEOUT=30m "$@" bash "$HERE/../ppo_pilot_job.sh"
}
# The synthetic start is not the D7 checkpoint, so the default id carries its sha prefix.
ID="pilot-${SHA:0:7}-s${CKPT_SHA:0:8}"
DEST="$MOCK_GCS/mock-bucket/model9/r5/ppo-pilot/$ID"
WORK="$SCRATCH/home/r5/ppo-pilot/$ID"

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

echo "=== D: tempered start checkpoint + head entropy floor"
win() { cygpath -w "$1" 2>/dev/null || echo "$1"; }
BUCKET="$MOCK_GCS/mock-bucket/model9/r5"
REF=$(ls "$DEST"/iter-000/*-rollout.pt | head -1)   # the uploaded iteration-0 rollout
TEMPERED="$SCRATCH/tempered-start.pt"
(cd "$REPO" && PYTHONPATH=src "$MOCK_VENV_PY" -m anthropocene_rl.autonomous_v7.labor_head_temper \
  --checkpoint "$(win "$BUCKET/d7-run/checkpoint-start.pt")" --reference-rollout "$(win "$REF")" \
  --fission-temperature 2 --trade-share-explore 0.2 --trade-share-log-std -1.5 \
  --out "$(win "$TEMPERED")" --report "$(win "$SCRATCH/temper.json")" > /dev/null)
TSHA=$(sha256sum "$TEMPERED" | cut -d' ' -f1)
mkdir -p "$BUCKET/ppo-pilot/tempered"
cp "$TEMPERED" "$BUCKET/ppo-pilot/tempered/checkpoint-start-tempered.pt"
TEMPERED_GCS=gs://mock-bucket/model9/r5/ppo-pilot/tempered/checkpoint-start-tempered.pt
if run_job CKPT_SHA256="$TSHA" START_CKPT_GCS="$TEMPERED_GCS" HEAD_ENTROPY_FLOOR="specialty=1:1" \
    > "$SCRATCH/D0.out" 2>&1; then echo "D: a labor-head floor should be refused"; exit 1; fi
grep -q "does not parse" "$SCRATCH/D0.out"
FLOOR="fission=1.2:0.05,trade_share=0:0.05"
FID="pilot-${SHA:0:7}-s${TSHA:0:8}-f$(printf %s "$FLOOR" | sha256sum | cut -c1-8)"
run_job CKPT_SHA256="$TSHA" START_CKPT_GCS="$TEMPERED_GCS" HEAD_ENTROPY_FLOOR="$FLOOR" JOB_TIMEOUT= \
  > "$SCRATCH/D.out" 2>&1 || { tail -30 "$SCRATCH/D.out"; exit 1; }
grep -q "pilot=$FID " "$SCRATCH/D.out"
TURNS_USED="${MOCK_TURNS:-2}"
if [ "$TURNS_USED" -le 8 ]; then CAP=7h; else CAP="$(( (7 * TURNS_USED + 7) / 8 ))h"; fi
grep -q "turns=$TURNS_USED .*cap=$CAP" "$SCRATCH/D.out"
grep -q "head tempering: \[{" "$SCRATCH/D.out"
[ -e "$BUCKET/ppo-pilot/$FID/EXIT_STATUS" ]
echo "D ok: $(cat "$BUCKET/ppo-pilot/$FID/EXIT_STATUS")"
(cd "$REPO" && PYTHONPATH=src "$MOCK_VENV_PY" - "$(win "$SCRATCH/home/r5/ppo-pilot/$FID")" "$TSHA" \
  "$(win "$SCRATCH/temper.json")" "$TURNS_USED" <<'PY'
import json, sys
from pathlib import Path
import torch
work, start_sha, turns = Path(sys.argv[1]), sys.argv[2], int(sys.argv[4])
for k in (0, 1):
    rollout = torch.load(work / f"iter-{k:03d}" / "rollout.pt", weights_only=True)
    assert rollout["metadata"]["turns"] == turns, rollout["metadata"]["turns"]
temper = json.loads(Path(sys.argv[3]).read_text())
assert abs(temper["achievedAfter"]["tradeShareExpected"] - 0.2) < 1e-3, temper["achievedAfter"]
for k in (0, 1):
    s = json.loads((work / f"iter-{k:03d}" / "summary.json").read_text())
    floor = s["headEntropyFloor"]
    assert set(floor["floors"]) == {"fission", "trade_share"}, floor
    assert set(s["headEntropy"]["in"]) == set(floor["entropyBefore"]), s["headEntropy"]
    update = torch.load(work / f"iter-{k:03d}" / "checkpoint-out.pt", weights_only=True)["last_update"]
    assert update["actor"]["headEntropyFloor"]["floorLossPerAttemptedStep"], update["actor"]
    if k == 0:
        assert s["sha256"]["checkpointIn"] == start_sha
    print(k, {"accepted": s["actor"]["acceptedSteps"], "fissionH": (
        round(s["headEntropy"]["in"]["fission"], 4), round(s["headEntropy"]["out"]["fission"], 4)),
        "shareH": (round(s["headEntropy"]["in"]["trade_share"], 4),
                   round(s["headEntropy"]["out"]["trade_share"], 4))})
PY
)
echo "MOCK E2E PASSED"
