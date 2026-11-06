# Single poller. Prints a heartbeat line every check; exits only on an ALERT (incl. batchdone), VM down > 150 min, or SSH failing while RUNNING.
cd "C:/Users/suyas/AppData/Local/Temp/claude/C--Users-suyas-Documents-ChatGPT-Anthropocene/0cac0cc5-1bf8-4f14-ae95-34de833627e0/scratchpad/r5"
SSH="gcloud compute ssh anthropocene-trainer-60 --project=anthropocene-508912 --zone=us-central1-c --tunnel-through-iap --strict-host-key-checking=yes --quiet"
st() { gcloud compute instances describe anthropocene-trainer-60 --project=anthropocene-508912 --zone=us-central1-c --format="value(status)" 2>&1; }
T() { echo "$(date -u +%FT%TZ) $*" >> tally.log; }
n=0; down=0
while true; do
  s=$(st)
  if [ "$s" != RUNNING ]; then
    [ $down -eq 0 ] && { down=$(date +%s); T "VM $s observed"; echo "EXIT: VM $s at $(date -u +%T)"; exit 0; }
    [ $(( $(date +%s) - down )) -gt 9000 ] && { echo "EXIT: VM NOT RESTARTED for 150 min (status $s)"; exit 0; }
    sleep 300; continue
  fi
  [ $down -ne 0 ] && { T "VM RUNNING again"; down=0; }
  out=$($SSH "--command=bash ~/r5/alert.sh" 2>&1)
  echo "$(date -u +%FT%TZ) $(echo "$out" | tr '\n' '|' | cut -c1-300)" >> poll.log
  if ! echo "$out" | grep -q 'TAILS='; then n=$((n+1)); echo "HB $(date -u +%T) ssh fail $n"; [ $n -ge 4 ] && { echo "EXIT: SSH FAILING while RUNNING"; exit 0; }; sleep 120; continue; fi; n=0
  echo "HB $(date -u +%T) $(echo "$out" | grep -E 'TAILS=|LOAD=' | tr '\n' ' ')"
  if echo "$out" | grep -q 'ALERT'; then echo "$out" | grep ALERT | while read l; do T "$l"; done; echo "EXIT: $(echo "$out" | grep ALERT | tr '\n' ';')"; exit 0; fi
  t=$(echo "$out" | sed -n 's/.*TAILS=\([0-9]*\).*/\1/p')
  sleep 1200   # 20 min checks while the 8-hourly guardrail stop is expected
done
