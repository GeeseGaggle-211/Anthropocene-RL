#!/bin/bash
# r5 curated Neolithic scenario bank: coverage-driven gap filling (manifest section 11).
#
# Stages (each skipped when its output is published; rerun the same command after a preemption):
#   1. coverage  -- labor_scenarios coverage over the 50-decision fitting and calibration banks
#                   (FIT_BANK, CAL_BANK) -> $D/coverage.json and coverage.txt;
#   2. collect   -- per partition (PARTITIONS), harvest roots only for the families below quota
#                   in that audit, record each root's teacher scenario with the randomized tail,
#                   one shard per world (--resume keeps verified shards) -> $D/<partition>;
#                   states recorded by the partition's bank (same reset options) are never roots;
#   3. report    -- labor_scenarios cost per partition and coverage-after.json (banks plus the
#                   new scenario sets);
#   4. search    -- optional (SEARCH=1): labor_search --source scenario on the fitting roots
#                   (A pairs) and, with HELD_OUT=1, on the reserved-world roots (--held-out-test).
#
# Partitions: fitting 90100-90153,90160-90165 (training), calibration 90180-90199 (calibration
# scenario set: report-only per-family calibration via SCENARIO_CAL_DATA in
# tools/r5_critic_acceptance.sh), held-out-test 90154-90159 (only with HELD_OUT=1). Never sealed
# 3000-3059 or development 2000-2059; labor_scenarios refuses them.
#
# Required: NORMALIZER (r5 actor normalizer), FIT_BANK, CAL_BANK. Optional: RESERVED_BANK.
# Usage: NORMALIZER=... FIT_BANK=... CAL_BANK=... tools/r5_scenarios.sh
set -u

H=${H:-$HOME}
ROOT=${SCEN_ROOT:-$H/r5}
D=${SCEN_DIR:-$ROOT/scenarios}
CODE=${SCEN_CODE:-$H/code/r5-scenarios/rl}
PY=${SCEN_PY:-$H/m9-bench/venv/bin/python}
: "${NORMALIZER:?}" "${FIT_BANK:?}" "${CAL_BANK:?}"
RESERVED_BANK=${RESERVED_BANK:-}
PARTITIONS=${PARTITIONS:-fitting calibration}
HELD_OUT=${HELD_OUT:-0}
FIT_WORLDS=${FIT_WORLDS:-90100-90153,90160-90165}
CAL_WORLDS=${CAL_WORLDS:-90180-90199}
RESERVED_WORLDS=${RESERVED_WORLDS:-90154-90159}
MINIMA=${MINIMA:-fitting=32/4,calibration=16/4,held-out-test=16/4}
DECISIONS=${DECISIONS:-100}            # harvest drive; a same-reset bank covers the first 50
HARVEST_START=${HARVEST_START:-0}
SCENARIO_DECISIONS=${SCENARIO_DECISIONS:-25}
OVERSAMPLE=${OVERSAMPLE:-1.5}
MAX_PER_WORLD=${MAX_PER_WORLD:-8}
ROOT_KEY=${ROOT_KEY:-0}
PLAYERS=${PLAYERS:-16}
SIZE=${SIZE---width 128 --height 64 --separation 9}
JOBS=${JOBS:-30}
SEARCH=${SEARCH:-0}
SEARCH_ARGS=${SEARCH_ARGS:-}
EXTRA=${EXTRA:-}                         # passed to every collect (e.g. --tail-draws ...)
DIRTY=${SCEN_ALLOW_DIRTY:+--allow-dirty}

[ -n "${ANTHROPOCENE_MODEL9_BRIDGE:-}" ] || export ANTHROPOCENE_MODEL9_BRIDGE=$CODE/integration/bin/model9/Anthropocene.Model9Bridge.dll
[ -n "${ANTHROPOCENE_DOTNET:-}" ] || { [ -x /usr/bin/dotnet ] && export ANTHROPOCENE_DOTNET=/usr/bin/dotnet; }
export PYTHONPATH=$CODE/src
mkdir -p "$D"
log() { echo "$(date -Is) $*" | tee -a "$D/driver.log"; }
fail() { log "FAIL $*"; exit 1; }
cd "$CODE" || fail "no learner checkout at $CODE"
for bank in "$FIT_BANK" "$CAL_BANK" ${RESERVED_BANK:+"$RESERVED_BANK"}; do
  [ -f "$bank/dataset.json" ] || fail "bank missing: $bank"
done
[ "$HELD_OUT" = "1" ] && PARTITIONS="$PARTITIONS held-out-test"
banks=(--dataset "$FIT_BANK" --dataset "$CAL_BANK")
[ -n "$RESERVED_BANK" ] && banks+=(--dataset "$RESERVED_BANK")
log "r5 scenarios start partitions=[$PARTITIONS] decisions=$DECISIONS K=$SCENARIO_DECISIONS players=$PLAYERS commit=$(git -C "$CODE" rev-parse --short HEAD 2>/dev/null)"

# 1. Coverage audit of the existing banks.
if [ ! -f "$D/coverage.json" ]; then
  "$PY" -m anthropocene_rl.autonomous_v7.labor_scenarios coverage "${banks[@]}" --minima "$MINIMA" \
    --out "$D/coverage.json.tmp" > "$D/coverage.txt" 2>> "$D/driver.log" || fail "coverage audit"
  mv "$D/coverage.json.tmp" "$D/coverage.json"
  log "coverage audit done"; cat "$D/coverage.txt" | tee -a "$D/driver.log"
fi

# 2. Gap-filling collection per partition.
outputs=()
for partition in $PARTITIONS; do
  case $partition in
    fitting) worlds=$FIT_WORLDS; bank=$FIT_BANK; flags= ;;
    calibration) worlds=$CAL_WORLDS; bank=$CAL_BANK; flags= ;;
    held-out-test) worlds=$RESERVED_WORLDS; bank=$RESERVED_BANK; flags=--held-out-test ;;
    *) fail "unknown partition $partition" ;;
  esac
  out=$D/$partition
  outputs+=("$out")
  if [ -f "$out/summary.json" ]; then log "$partition already collected"; continue; fi
  exclude=()
  [ -n "$bank" ] && exclude=(--exclude-dataset "$bank")
  resume=; [ -d "$out" ] && resume=--resume
  log "$partition collect worlds=$worlds $resume"
  # shellcheck disable=SC2086
  "$PY" -m anthropocene_rl.autonomous_v7.labor_scenarios collect \
    --normalizer-checkpoint "$NORMALIZER" --partition "$partition" $flags --worlds "$worlds" \
    --coverage "$D/coverage.json" "${exclude[@]}" --decisions "$DECISIONS" \
    --harvest-start "$HARVEST_START" --scenario-decisions "$SCENARIO_DECISIONS" \
    --oversample "$OVERSAMPLE" --max-per-world "$MAX_PER_WORLD" --root-key "$ROOT_KEY" \
    --players "$PLAYERS" $SIZE --jobs "$JOBS" $resume $DIRTY $EXTRA --out "$out" \
    >> "$D/driver.log" 2>> "$D/$partition.stderr.log" || fail "$partition collection"
  log "$partition collected"
done

# 3. Cost and post-fill coverage.
for out in "${outputs[@]}"; do
  [ -f "$out/cost.json" ] || "$PY" -m anthropocene_rl.autonomous_v7.labor_scenarios cost \
    --dataset "$out" > "$out/cost.json" 2>> "$D/driver.log"
done
if [ ! -f "$D/coverage-after.json" ]; then
  after=("${banks[@]}")
  for out in "${outputs[@]}"; do [ -f "$out/dataset.json" ] && after+=(--dataset "$out"); done
  "$PY" -m anthropocene_rl.autonomous_v7.labor_scenarios coverage "${after[@]}" --minima "$MINIMA" \
    --out "$D/coverage-after.json.tmp" > "$D/coverage-after.txt" 2>> "$D/driver.log" \
    && mv "$D/coverage-after.json.tmp" "$D/coverage-after.json"
  log "post-fill coverage"; cat "$D/coverage-after.txt" | tee -a "$D/driver.log"
fi

# 4. Optional search on scenario roots (A pairs; reserved roots only as held-out tests).
if [ "$SEARCH" = "1" ]; then
  for partition in fitting held-out-test; do
    roots=$D/$partition/roots.json
    [ -f "$roots" ] || continue
    out=$D/search-$partition
    [ -f "$out/dataset.json" ] && { log "search $partition already done"; continue; }
    rm -rf "$out"  # labor_search never resumes a partial output
    case $partition in fitting) worlds=$FIT_WORLDS; flags= ;; *) worlds=$RESERVED_WORLDS; flags=--held-out-test ;; esac
    log "search $partition on scenario roots"
    # shellcheck disable=SC2086
    "$PY" -m anthropocene_rl.autonomous_v7.labor_search --normalizer-checkpoint "$NORMALIZER" \
      --worlds "$worlds" --source scenario --scenario-roots "$roots" $flags \
      --players "$PLAYERS" $SIZE --jobs "$JOBS" $DIRTY $SEARCH_ARGS --out "$out" \
      >> "$D/driver.log" 2>> "$D/search-$partition.stderr.log" || fail "search $partition"
  done
fi
log "r5 scenarios done"
