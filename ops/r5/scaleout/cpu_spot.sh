#!/usr/bin/env bash
# Switch the CPU VM back to Spot before the search (user 28 Oct, Q4), from the OPERATOR's
# machine (gcloud with compute rights): a VM cannot change its own scheduling, which needs
# the instance stopped. Prints every command; DRY_RUN=0 runs them.
#
# Order: stop (if running) -> set-scheduling SPOT -> describe (verify) -> start (START=1).
# Kept as is: the 8-hour guardrail (a startup-script systemd timer inside the VM) and the
# max-run duration (MAX_RUN, default 40h, re-applied because set-scheduling rewrites the
# scheduling block). Spot requires --no-restart-on-failure and --maintenance-policy=TERMINATE;
# the termination action stays STOP so the disk and ~/r5 survive a preemption.
# If the API rejects the call with "preemptible must be true" (older API behaviour), add
# --preemptible (EXTRA="--preemptible").
set -euo pipefail
PROJECT="${PROJECT:-anthropocene-508912}"
ZONE="${ZONE:-us-central1-c}"
NAME="${NAME:-anthropocene-trainer-60}"
MAX_RUN="${MAX_RUN:-40h}"
START="${START:-1}"
DRY_RUN="${DRY_RUN:-1}"
EXTRA="${EXTRA:-}"

run() {
  printf '+'; printf ' %q' "$@"; printf '\n'
  if [ "$DRY_RUN" = "0" ]; then "$@" < /dev/null; fi
}
[ "$DRY_RUN" = "0" ] || echo "# DRY RUN (DRY_RUN=0 to execute)"

status=$(timeout 120 gcloud compute instances describe "$NAME" --zone "$ZONE" --project "$PROJECT" \
  --format='value(status)' < /dev/null)
echo "# $NAME status: $status"
if [ "$status" != "TERMINATED" ] && [ "$status" != "STOPPED" ]; then
  run timeout 600 gcloud compute instances stop "$NAME" --zone "$ZONE" --project "$PROJECT"
fi
# shellcheck disable=SC2086
run timeout 300 gcloud compute instances set-scheduling "$NAME" --zone "$ZONE" --project "$PROJECT" \
  --provisioning-model=SPOT --instance-termination-action=STOP --no-restart-on-failure \
  --maintenance-policy=TERMINATE --max-run-duration="$MAX_RUN" $EXTRA
run timeout 120 gcloud compute instances describe "$NAME" --zone "$ZONE" --project "$PROJECT" \
  --format='yaml(scheduling)'
if [ "$START" = "1" ]; then
  run timeout 600 gcloud compute instances start "$NAME" --zone "$ZONE" --project "$PROJECT"
fi
echo "# expected scheduling: provisioningModel SPOT, instanceTerminationAction STOP, automaticRestart false, onHostMaintenance TERMINATE, maxRunDuration $MAX_RUN"
