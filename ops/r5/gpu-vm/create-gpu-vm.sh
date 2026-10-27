#!/usr/bin/env bash
# Create the r5 GPU fitting VM (NVIDIA L4, Spot). PREPARED, NOT RUN.
#
# Review README.md first. Nothing here may run until the co-architect approves the plan
# and the GPU quota exists (GPUS_ALL_REGIONS is 0 today; see README "Quota").
#
# What it creates (each step is idempotent-ish and prints before acting):
#   1. service account  anthropocene-gpu-fit@  (no project roles)
#   2. bucket-level IAM roles/storage.objectUser on gs://anthropocene-rl-datasets-508912 only
#   3. the VM anthropocene-gpu-fit-l4: g2-standard-16 (1x L4, 16 vCPU, 64 GB), Spot,
#      termination action STOP, max run duration 20h, no external IP, IAP SSH tag,
#      Shielded VM, 200 GB pd-balanced boot disk, Ubuntu 24.04 accelerator image with the
#      NVIDIA 580 driver preinstalled, labels.
# It never touches anthropocene-trainer-60, firewall rules, routers or subnets.
#
# Usage (after approval):  DRY_RUN=0 ./create-gpu-vm.sh
# Default is DRY_RUN=1: print every command, run none.
set -euo pipefail

PROJECT="${PROJECT:-anthropocene-508912}"
ZONE="${ZONE:-us-central1-c}"            # L4 is offered in us-central1-a, -b, -c
NAME="${NAME:-anthropocene-gpu-fit-l4}"
MACHINE="${MACHINE:-g2-standard-16}"     # 64 GB host RAM: 4 concurrent ~11 GB critic fits
BUCKET="${BUCKET:-anthropocene-rl-datasets-508912}"
SA_ID="${SA_ID:-anthropocene-gpu-fit}"
SA="${SA_ID}@${PROJECT}.iam.gserviceaccount.com"
NETWORK="${NETWORK:-anthropocene-training}"
SUBNET="${SUBNET:-anthropocene-training-us-central1}"
IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-accelerator-images}"
IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-accelerator-2404-amd64-with-nvidia-580}"
DISK_GB="${DISK_GB:-200}"
DRY_RUN="${DRY_RUN:-1}"

run() {
  printf '+'; printf ' %q' "$@"; printf '\n'
  if [ "$DRY_RUN" = "0" ]; then "$@"; fi
}

if [ "$DRY_RUN" != "0" ]; then
  echo "# DRY RUN (set DRY_RUN=0 to execute after approval). Commands that would run:"
fi

# 0. Preconditions (read-only): quota and the network pieces a no-external-IP VM needs.
echo "# preflight (read-only)"
run gcloud compute project-info describe --project "$PROJECT"   --flatten=quotas --format="csv[no-heading](quotas.metric,quotas.usage,quotas.limit)"
run gcloud compute regions describe us-central1 --project "$PROJECT"   --flatten=quotas --format="csv[no-heading](quotas.metric,quotas.usage,quotas.limit)"

# 1. Minimal-scope service account (no project-level roles).
if ! gcloud iam service-accounts describe "$SA" --project "$PROJECT" >/dev/null 2>&1; then
  run gcloud iam service-accounts create "$SA_ID" --project "$PROJECT" \
    --display-name="Anthropocene r5 GPU fitting (GCS r/w on the dataset bucket only)"
fi

# 2. Object read/write on the one bucket. objectUser = get/list/create/overwrite/delete
#    objects; no bucket or IAM administration. Nothing at project level.
run gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${SA}" --role="roles/storage.objectUser"

# 3. The VM. The OAuth scope is limited to Cloud Storage read/write as a second fence.
run gcloud compute instances create "$NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --max-run-duration=20h \
  --maintenance-policy=TERMINATE \
  --no-restart-on-failure \
  --network="$NETWORK" \
  --subnet="$SUBNET" \
  --no-address \
  --tags=iap-ssh \
  --service-account="$SA" \
  --scopes="https://www.googleapis.com/auth/devstorage.read_write" \
  --image-project="$IMAGE_PROJECT" \
  --image-family="$IMAGE_FAMILY" \
  --boot-disk-size="${DISK_GB}GB" \
  --boot-disk-type=pd-balanced \
  --boot-disk-device-name="${NAME}-boot" \
  --shielded-secure-boot \
  --shielded-vtpm \
  --shielded-integrity-monitoring \
  --metadata=enable-oslogin=TRUE,block-project-ssh-keys=TRUE \
  --labels=environment=research,purpose=anthropocene-training,generation=l4-gpu,role=r5-fitting,managed-by=claude

# g2 machine types carry their L4 implicitly; no --accelerator flag is needed.
# The STOP action keeps the disk (data, venv, partial checkpoints) across preemptions;
# max-run-duration also STOPs after 20 h of run time, capping a forgotten VM at ~20 h of
# compute. A STOPPED VM still bills the 200 GB disk (~$0.66/day).

cat <<EOF
# next (after the VM exists):
#   gcloud compute ssh $NAME --zone $ZONE --project $PROJECT --tunnel-through-iap
#   then follow README.md "Bootstrap" (bootstrap.sh).
EOF
