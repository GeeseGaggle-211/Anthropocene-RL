# Prints ALERT lines for: low memory, OOM, errors in batch logs, driver FAIL/STOP, newly completed batches. Prints TAILS=<n>.
D=/home/suyashbhatt11_gmail_com/r5; S=$D/alert.state; touch $S
av=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
[ $av -lt 18 ] && echo "ALERT lowmem avail=${av}GB"
n=$(sudo -n journalctl -k --since "-25 min" 2>/dev/null | grep -ci 'oom-kill\|out of memory'); [ "$n" -gt 0 ] && echo "ALERT oom lines=$n"
for f in $D/fit8-b0?.log $D/cal8-b0?.log; do [ -f $f ] || continue; c=$(grep -c -E 'Traceback' $f); [ $c -gt 0 ] && echo "ALERT traceback in $(basename $f)"; done
grep -qE 'FAIL (batch|fit|priv)|incomplete' $D/driver.log && ! grep -q 'driverfail' $S && { echo "ALERT driver: $(grep -E 'FAIL (batch|fit|priv)|incomplete' $D/driver.log | tail -1)"; echo driverfail >> $S; }
[ -f $D/STOP ] && ! grep -q 'stop' $S && { echo "ALERT STOP present"; echo stop >> $S; }
for d in $D/fit8-b0? $D/cal8-b0? $D/heldout50; do n=$(basename $d); [ -f $d/dataset.json ] && ! grep -qx "$n" $S && { echo "ALERT batchdone $n"; echo $n >> $S; }; done
[ -f $D/COLLECT_DONE ] && ! grep -q collectdone $S && { echo "ALERT COLLECT_DONE"; echo collectdone >> $S; }
t=0; for f in $D/fit8-b0?.log $D/cal8-b0?.log; do [ -f $f ] || continue
 t=$((t + $(python3 - $f <<'PY'
import json,sys,os
f=sys.argv[1]; last={}
for l in open(f):
    if l.startswith('{"world"'):
        d=json.loads(l); last[d["world"]]=d
d=f[:-4]; pub={int(x[6:]) for x in os.listdir(d) if x.startswith("world-")} if os.path.isdir(d) else set()
print(sum(1 for w,x in last.items() if x["decisions"]>=50 and w not in pub))
PY
))); done
echo "TAILS=$t avail=${av}GB"
echo "LOAD=$(cut -d' ' -f1 /proc/loadavg) W=$(pgrep -fc 'autonomous_v7[.]labor_teacher_data')"
grep -E 'batch .* (start|done)|moved partial|driver start' /home/suyashbhatt11_gmail_com/r5/driver.log | tail -6
