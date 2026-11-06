#!/bin/bash
# Relaunch the r5 driver (and the memory watcher) every 5 minutes until COLLECT_DONE.
# The driver takes its own lock (~/r5/drive-main.lock) and exits at once if another driver holds it.
D=/home/suyashbhatt11_gmail_com/r5
while [ ! -f $D/COLLECT_DONE ]; do
  pgrep -f 'r5/memwatch[.]sh' > /dev/null || (setsid nohup bash $D/memwatch.sh > /dev/null 2>&1 < /dev/null &)
  bash $D/drive.sh >> $D/drive.out 2>&1
  sleep 300
done
