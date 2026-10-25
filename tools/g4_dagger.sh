#!/bin/bash
# G4 learner-state recovery (DAgger) driver for the labor actor (brief 6.1 and 6.4 S2).
#
# Rounds r = 1..ROUNDS with depth d = DEPTHS[r]. Each round:
#   1. collect learner-state rows with the current actor (labor_dagger collect, --resume safe);
#   2. refit the actor from the current checkpoint for EPOCHS epochs on the original teacher data
#      plus every aggregated round so far (pretrain-bc --dataset ... at 50/50 source mass), through
#      tools/g0launch.py so <out>.partial is written every G0_SAVE_EVERY steps;
#   3. evaluate (labor_pretrain_eval autonomous, modes h12,deterministic,stochastic) and log
#      welfare/person-years versus H and versus H12 to $D/driver.log.
# Idempotent and resumable after a Spot preemption: completed phases are skipped by their outputs
# and marker files, a collection resumes from its verified shards, and a refit resumes from its
# .partial with the remaining epochs (Adam moments restart). Critics are never trained here: the
# actor checkpoint keeps the critics of its input. After the last round $ROOT/pretrained-g4.pt
# combines the final actor with the critics of CRITIC (argument 1, default $ROOT/pretrained.pt).
#
# Usage: tools/g4_dagger.sh [CRITIC_CHECKPOINT]
# Every setting below can be overridden from the environment (see the local smoke test in
# tests/test_labor_dagger.py and docs in the commit message).
set -u

H=${H:-$HOME}
ROOT=${G4_ROOT:-$H/g0}                       # holds fit-data, pretrained.pt
D=${G4_DIR:-$ROOT/dagger}                    # round outputs and driver.log
CODE=${G4_CODE:-$H/code/g4-dagger/rl}        # learner checkout (this repository)
PY=${G4_PY:-$H/m9-bench/venv/bin/python}
FIT=${G4_FIT:-$ROOT/fit-data}
START=${G4_START:-$ROOT/pretrained.pt}       # actor checkpoint entering round 1
CRITIC=${1:-${G4_CRITIC:-$ROOT/pretrained.pt}}
WORLDS=${G4_WORLDS:-90100-90159}
ROUNDS=${G4_ROUNDS:-5}
DEPTHS=(${G4_DEPTHS:-1 2 3 4 5})
STARTS=${G4_STARTS:-3}
HORIZON=${G4_HORIZON:-25}
ACTOR_MODE=${G4_ACTOR_MODE:-}                # empty = deterministic means; "--stochastic" to sample
JOBS=${G4_JOBS:-30}
PLAYERS=${G4_PLAYERS:-16}
SIZE=${G4_SIZE---width 128 --height 64 --separation 9}
EPOCHS=${G4_EPOCHS:-2}
BATCH=${G4_BATCH:-256}
LR=${G4_LR:-5e-4}
THREADS=${G4_THREADS:-28}
EVAL_WORLDS=${G4_EVAL_WORLDS:-2000-2007}
EVAL_TURNS=${G4_EVAL_TURNS:-8}
EVAL_JOBS=${G4_EVAL_JOBS:-24}
EVAL_MODES=${G4_EVAL_MODES:-h12,deterministic,stochastic}
DIRTY=${G4_ALLOW_DIRTY:+--allow-dirty}

[ -n "${ANTHROPOCENE_MODEL9_BRIDGE:-}" ] || export ANTHROPOCENE_MODEL9_BRIDGE=$CODE/integration/bin/model9/Anthropocene.Model9Bridge.dll
[ -n "${ANTHROPOCENE_DOTNET:-}" ] || { [ -x /usr/bin/dotnet ] && export ANTHROPOCENE_DOTNET=/usr/bin/dotnet; }
export PYTHONPATH=$CODE/src
mkdir -p "$D"
log() { echo "$(date -Is) $*" | tee -a "$D/driver.log"; }
fail() { log "FAIL $*"; exit 1; }
cd "$CODE" || fail "no learner checkout at $CODE"

[ -f "$FIT/dataset.json" ] || fail "teacher dataset missing: $FIT"
[ -f "$START" ] || fail "start checkpoint missing: $START"

partial_steps() {
  "$PY" -c "import torch,sys; d=torch.load(sys.argv[1],map_location='cpu',weights_only=False); print(int((d.get('g0_partial') or {}).get('steps',0)))" "$1" 2>/dev/null || echo 0
}
dataset_rows() {  # total rows over the given published dataset directories
  "$PY" -c "import json,sys; print(sum(s['rows'] for p in sys.argv[1:] for s in json.load(open(p+'/dataset.json'))['shards'].values()))" "$@"
}

log "g4 start rounds=$ROUNDS depths=${DEPTHS[*]} starts=$STARTS epochs=$EPOCHS worlds=$WORLDS start=$START critic=$CRITIC commit=$(git -C "$CODE" rev-parse --short HEAD 2>/dev/null)"
current=$START
datasets=("$FIT")
for ((r = 1; r <= ROUNDS; r++)); do
  depth=${DEPTHS[$((r - 1))]}
  R=$D/round-$r
  mkdir -p "$R"
  if [ -f "$D/round-$r.done" ]; then
    datasets+=("$R/data"); current=$R/actor.pt
    log "round $r already done"
    continue
  fi

  # 1. Learner-state collection with the frozen current actor.
  if [ ! -f "$R/data/dataset.json" ]; then
    log "round $r collect depth=$depth checkpoint=$current"
    resume=""; [ -d "$R/data" ] && resume="--resume"
    OMP_NUM_THREADS=1 "$PY" -m anthropocene_rl.autonomous_v7.labor_dagger collect \
      --checkpoint "$current" --worlds "$WORLDS" --round "$r" --depth "$depth" --starts "$STARTS" \
      --horizon "$HORIZON" --players "$PLAYERS" $SIZE --jobs "$JOBS" $ACTOR_MODE $resume $DIRTY \
      --out "$R/data" >> "$R/collect.log" 2>&1 || fail "round $r collection (see $R/collect.log)"
    log "round $r collected $(tail -n 1 "$R/collect.log")"
  fi
  datasets+=("$R/data")

  # 2. Actor refit on the teacher data plus every aggregated round (critics untouched).
  if [ ! -f "$R/actor.pt" ]; then
    rows=$(dataset_rows "${datasets[@]}")
    spe=$(( (rows + BATCH - 1) / BATCH ))
    epochs=$EPOCHS; ck=$current
    if [ -f "$R/actor.pt.partial" ]; then
      done_steps=$(partial_steps "$R/actor.pt.partial")
      cp "$R/actor.pt.partial" "$R/actor.resume.pt"; ck=$R/actor.resume.pt
      epochs=$(( EPOCHS - done_steps / spe )); [ $epochs -lt 1 ] && epochs=1
      log "round $r refit resume from partial at $done_steps steps ($spe/epoch); $epochs epochs left"
    else
      log "round $r refit $epochs epochs rows=$rows steps/epoch=$spe from $current"
    fi
    args=(); for ds in "${datasets[@]}"; do args+=(--dataset "$ds"); done
    OMP_NUM_THREADS=$THREADS "$PY" "$CODE/tools/g0launch.py" pretrain-bc --checkpoint "$ck" "${args[@]}" \
      --epochs "$epochs" --batch-size "$BATCH" --lr "$LR" --max-grad-norm 1.0 --allow-vm-scale \
      --out "$R/actor.pt" >> "$R/refit.log" 2>&1 || fail "round $r refit (see $R/refit.log)"
    rm -f "$R/actor.resume.pt"
    log "round $r refit done"
  fi
  current=$R/actor.pt

  # 3. Fully autonomous development evaluation (no teacher handoff).
  if [ ! -f "$R/eval.json" ]; then
    log "round $r eval"
    OMP_NUM_THREADS=1 "$PY" -m anthropocene_rl.autonomous_v7.labor_pretrain_eval autonomous \
      --checkpoint "$current" --worlds "$EVAL_WORLDS" --turns "$EVAL_TURNS" --modes "$EVAL_MODES" \
      --players "$PLAYERS" $SIZE --jobs "$EVAL_JOBS" --out "$R/eval.json" \
      >> "$R/eval.log" 2>&1 || fail "round $r eval (see $R/eval.log)"
  fi
  summary=$("$PY" - "$R/eval.json" "$R/data/dataset.json" <<'EOF'
import json, sys
rep = json.load(open(sys.argv[1]))
rows = sum(s["rows"] for s in json.load(open(sys.argv[2]))["shards"].values())
agg = rep["aggregate"]
out = [f"rows={rows}"]
h12 = agg.get("h12", {}).get("actor")
for mode in ("h12", "deterministic", "stochastic"):
    if mode not in agg:
        continue
    a, h = agg[mode]["actor"], agg[mode]["heuristic"]
    part = f"{mode}: W/H={a['welfare'] / h['welfare']:.3f} PY/H={a['person_years'] / h['person_years']:.3f}"
    if h12 and mode != "h12":
        part += f" W/H12={a['welfare'] / h12['welfare']:.3f} PY/H12={a['person_years'] / h12['person_years']:.3f}"
    out.append(part + f" gate={agg[mode]['gate']}")
print(" | ".join(out))
EOF
  ) || fail "round $r summary"
  echo "$summary" > "$R/summary.txt"
  touch "$D/round-$r.done"
  log "round $r done depth=$depth $summary"
done

# Final combined checkpoint: last actor + CRITIC's critics.
FINAL=$D/round-$ROUNDS/actor.pt
if [ ! -f "$ROOT/pretrained-g4.pt" ]; then
  "$PY" - "$FINAL" "$CRITIC" "$ROOT/pretrained-g4.pt" <<'EOF' >> "$D/driver.log" 2>&1 || fail "combine"
import os, sys, torch
actor_path, critic_path, out_path = sys.argv[1:]
actor = torch.load(actor_path, map_location="cpu", weights_only=False)
critic = torch.load(critic_path, map_location="cpu", weights_only=False)
for key in ("normalizer", "labor_model_contract", "feature_contract", "model_input_contract", "labor_encoder_layout"):
    assert str(actor.get(key)) == str(critic.get(key)), key
out = dict(actor)
for key in ("task_critic", "cost_critics", "critic_optimizer", "critic_training_world_seeds",
            "critic_training_independent_unit_ids", "heuristic_critic_pretraining"):
    if key in critic:
        out[key] = critic[key]
out["g4_combination"] = {"actor_from": actor_path, "critics_from": critic_path}
torch.save(out, out_path + ".tmp"); os.replace(out_path + ".tmp", out_path)
print("combined", out_path)
EOF
fi
log "g4 complete final_actor=$FINAL combined=$ROOT/pretrained-g4.pt"
