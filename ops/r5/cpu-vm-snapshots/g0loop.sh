#!/bin/bash
# Non-persistent driver: run the idempotent pipeline every 5 minutes until it marks ALLDONE.
# Lives only until the VM stops; after a restart, launch it again (one command).
while [ ! -f $HOME/g0/ALLDONE ]; do
  bash $HOME/g0pipe.sh >> $HOME/g0/cron.log 2>&1
  sleep 300
done
