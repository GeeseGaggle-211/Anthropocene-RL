#!/bin/bash
# Samples every 120 s: per collector worker (python child of a labor_teacher_data parent) its own RSS and its process-tree RSS (incl. dotnet bridges), plus MemAvailable.
D=/home/suyashbhatt11_gmail_com/r5; O=$D/memwatch.csv
[ -f $O ] || echo "utc,pid,etime_s,rss_self_mb,rss_tree_mb,mem_avail_gb" > $O
tree_rss() { local p=$1 s; s=$(ps -o rss= -p $p 2>/dev/null); s=${s:-0}; for c in $(pgrep -P $p); do s=$((s + $(tree_rss $c))); done; echo $s; }
while [ ! -f $D/COLLECT_DONE ]; do
  t=$(date -u +%FT%TZ); av=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
  for par in $(pgrep -f 'autonomous_v7[.]labor_teacher_data'); do
    pp=$(ps -o ppid= -p $par | tr -d ' ')
    # workers: python processes whose parent is a labor_teacher_data python
    if ps -o args= -p $pp | grep -q 'labor_teacher_data'; then
      echo "$t,$par,$(ps -o etimes= -p $par | tr -d ' '),$(( $(ps -o rss= -p $par) / 1024 )),$(( $(tree_rss $par) / 1024 )),$av" >> $O
    fi
  done
  echo "$t,total,,,$(( $(ps -eo rss= | awk '{s+=$1} END {print s}') / 1024 )),$av" >> $O
  sleep 120
done
