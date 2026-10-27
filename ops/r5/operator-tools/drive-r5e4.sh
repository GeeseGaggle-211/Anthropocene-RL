#!/bin/bash
# r5 collection driver (idempotent). Run as: bash ~/r5/drive.sh (from ~/r5/loop.sh).
# The driver takes its own lock (fd 9 on ~/r5/drive-main.lock); batch subshells close fd 9, so killing the
# driver frees the lock while its batches keep running. in_use() prevents a relaunch from duplicating a running batch.
# Collection and the approved post-collection commands use the r5e learner (checkpoint every decision, exact --resume).
H=/home/suyashbhatt11_gmail_com; D=$H/r5; C=$H/code/r5e/rl; PY=$H/m9-bench/venv/bin/python
export PATH=/snap/bin:/usr/local/bin:/usr/bin:/bin:$PATH
export ANTHROPOCENE_DOTNET=/usr/bin/dotnet ANTHROPOCENE_MODEL9_BRIDGE=$C/integration/bin/model9/Anthropocene.Model9Bridge.dll PYTHONPATH=$C/src OMP_NUM_THREADS=1
M=anthropocene_rl.autonomous_v7
log() { echo "$(date -Is) $*" >> $D/driver.log; }
[ -f $D/COLLECT_DONE ] && exit 0
[ -f $D/STOP ] && exit 0   # set on a hard failure; the operator clears it
exec 9>$D/drive-main.lock; flock -n 9 || exit 0
cd $C
log "driver start pid=$$ rl=$(git -C $C rev-parse --short HEAD)"

# Step 2: actor normalizer on fitting worlds 90100-90107.
if [ ! -f $D/actor-normalizer.pt ]; then
  log "fit-normalizer start"
  $PY -m $M.labor_training fit-normalizer --worlds 90100,90101,90102,90103,90104,90105,90106,90107 --turns 25 \
    --players 16 --width 128 --height 64 --separation 9 --jobs 8 --allow-vm-scale --out $D/actor-normalizer.pt >> $D/actor-normalizer.log 2>&1 \
    || { log "FAIL fit-normalizer (see actor-normalizer.log)"; touch $D/STOP; exit 1; }
  log "fit-normalizer done sha256=$(sha256sum $D/actor-normalizer.pt | cut -c1-16)"
fi

[ -f $D/JOBS ] && R5_JOBS=$(cat $D/JOBS)
COMMON="--normalizer-checkpoint $D/actor-normalizer.pt --turns 50 --players 16 --width 128 --height 64 --separation 9 --personality dirichlet:1.0 --personality-key 4242 --return-tail randomized --jobs ${R5_JOBS:-8}"
BATCHES="fit8-b01:90100-90107 fit8-b02:90108-90115 fit8-b03:90116-90123 fit8-b04:90124-90131 fit8-b05:90132-90139 fit8-b06:90140-90147 fit8-b07:90148-90153,90160-90161 fit8-b08:90162-90165 cal8-b01:90180-90187 cal8-b02:90188-90195 cal8-b03:90196-90199 heldout50:90154-90159"

run_batch() {
  local name=$1 worlds=$2 extra="" resume=
  [ $name = heldout50 ] && extra="--held-out-test"
  # An existing output dir is continued in place from its checkpoints; it is never moved or removed.
  [ -d "$D/$name" ] && resume=--resume
  log "batch $name start worlds=$worlds $extra $resume"
  if $PY -m $M.labor_teacher_data $COMMON $extra --worlds $worlds --out $D/$name $resume >> $D/$name.log 2>&1 && [ -f $D/$name/dataset.json ]; then
    log "batch $name done"
  else
    log "FAIL batch $name rc=$? (see $name.log)"; touch $D/$name.FAILED
  fi
}

# Concurrency is decided by active collector workers, not by batch count (co-architect rule, 27 Oct):
# start the next batch when batch_parents < MAXB AND active workers <= MAXW AND MemAvailable >= MINMEM GB
# (27 Oct revision: 3 / 10 / 60). ~/r5/launch.conf, if present, is re-read before every check, so the rule
# can be changed without restarting the driver.
# A batch parent is a labor_teacher_data process whose parent is not one; a worker is one whose parent is.
# A worker is "active" if it used CPU during a 10 s sample (idle pool workers of a draining batch do not count).
collectors() { pgrep -f 'autonomous_v7[.]labor_teacher_data'; }
batch_parents() { local n=0 p; for p in $(collectors); do ps -o args= -p $(ps -o ppid= -p $p) 2>/dev/null | grep -q 'labor_teacher_data' || n=$((n+1)); done; echo $n; }
active_workers() {
  local p a b n=0; declare -A t0
  for p in $(collectors); do ps -o args= -p $(ps -o ppid= -p $p) 2>/dev/null | grep -q 'labor_teacher_data' && t0[$p]=$(ps -o times= -p $p | tr -d ' '); done
  sleep 10
  for p in "${!t0[@]}"; do b=$(ps -o times= -p $p 2>/dev/null | tr -d ' '); [ -n "$b" ] && [ "$b" -gt "${t0[$p]}" ] && n=$((n+1)); done
  echo $n
}
mem_gb() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
in_use() { pgrep -f -- "labor_teacher_data .*--out $D/$1( |\$)" > /dev/null; }

for b in $BATCHES; do
  name=${b%%:*}; worlds=${b#*:}
  [ -f $D/$name/dataset.json ] && continue
  [ -f $D/$name.FAILED ] && { log "skip $name: FAILED marker present"; continue; }
  in_use $name && { log "skip $name: a collector is already writing $D/$name"; continue; }
  while true; do
    MAXB=3; MAXW=10; MINMEM=60; [ -f $D/launch.conf ] && . $D/launch.conf
    bp=$(batch_parents); aw=$(active_workers); mg=$(mem_gb)
    [ $bp -lt $MAXB ] && [ $aw -le $MAXW ] && [ $mg -ge $MINMEM ] && break
    sleep 50
  done
  log "launch $name: batches=$bp active_workers=$aw mem=${mg}GB"
  run_batch $name $worlds 9>&- &
  sleep 60
done
wait
# also wait for any batch this driver did not start (e.g. one inherited from an earlier driver)
while [ $(batch_parents) -gt 0 ]; do sleep 300; done

for b in $BATCHES; do name=${b%%:*}; [ -f $D/$name/dataset.json ] || { log "collection incomplete: $name missing; stop"; touch $D/STOP; exit 1; }; done
log "all collections published"
# post-collection stage runs only after the co-architect approves its exact commands
[ -f $D/POST_APPROVED ] || { log "post-collection stage waiting for ~/r5/POST_APPROVED"; exit 0; }

# Step 4: privileged normalizer (fitting batches only; calibration excluded) and dead-mask audit.
FITS=""; for n in fit8-b01 fit8-b02 fit8-b03 fit8-b04 fit8-b05 fit8-b06 fit8-b07 fit8-b08; do FITS="$FITS --dataset $D/$n"; done
if [ ! -f $D/privileged-normalizer.pt ]; then
  log "fit-privileged-normalizer start"
  (OMP_NUM_THREADS=8 $PY -m $M.labor_training fit-privileged-normalizer $FITS --calibration-dataset $D/cal8-b01 --calibration-dataset $D/cal8-b02 --calibration-dataset $D/cal8-b03 \
    --report $D/privileged-normalizer-report.json --out $D/privileged-normalizer.pt >> $D/privileged-normalizer.log 2>&1) \
    || { log "FAIL fit-privileged-normalizer"; touch $D/STOP; exit 1; }
  log "fit-privileged-normalizer done: $(tail -1 $D/privileged-normalizer.log | cut -c1-400)"
fi
if [ ! -f $D/privileged-dead-mask-audit.json ]; then
  (OMP_NUM_THREADS=8 $PY -m $M.labor_training privileged-dead-mask-audit $FITS --out $D/privileged-dead-mask-audit.json >> $D/privileged-audit.log 2>&1) \
    || { log "FAIL privileged-dead-mask-audit"; touch $D/STOP; exit 1; }
  log "dead-mask audit done: $(tail -1 $D/privileged-audit.log | cut -c1-400)"
fi
touch $D/COLLECT_DONE
log "COLLECT_DONE"
# idle rule (approved 27 Oct): stop the VM once collection and the post-collection stage are complete
log "stopping the VM: r5 collection complete"
sudo -n shutdown -h now "r5 collection complete"
