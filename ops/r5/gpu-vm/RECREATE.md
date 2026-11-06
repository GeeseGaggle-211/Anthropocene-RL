# Recreating anthropocene-gpu-fit-l4 (handoff, 26-27 Oct 2026)

State: the instance is DELETED. Nothing is running. The boot disk was kept.
Why: it was a Spot VM, preempted twice in about 2.5 h against 3+ h collection jobs. The in-place Spot-to-standard conversion needs `--no-preemptible` (found later), so I deleted and tried to recreate it as STANDARD. That failed with a us-central1-c L4 STOCKOUT (18 attempts over about 20 min). Collection moved back to the CPU VM.

## What exists
- Boot disk: `anthropocene-gpu-fit-l4` (us-central1-c, 200 GB pd-balanced, auto-delete off).
- Snapshot: `anthropocene-gpu-fit-l4-boot-20261026` (READY, storage us-central1, about 8 GB). It can be restored in any zone.
- Saved config: `gpu-vm-before.yaml` beside this file (full `describe` of the Spot instance; it contains shielded-boot blobs).
- Dev account: `anthropocene-gpu-fit@anthropocene-508912.iam.gserviceaccount.com` (storage.objectUser on the bucket only) is unchanged.

## Recreate (same zone, existing disk)
Common settings come from create-gpu-vm.sh. Only the disk flag and the provisioning flags differ.

STANDARD (on-demand):
```bash
N=anthropocene-gpu-fit-l4
gcloud compute instances create $N --project=anthropocene-508912 --zone=us-central1-c \
  --machine-type=g2-standard-16 --provisioning-model=STANDARD --maintenance-policy=TERMINATE \
  --max-run-duration=20h --instance-termination-action=STOP --no-restart-on-failure \
  --network=anthropocene-training --subnet=anthropocene-training-us-central1 --no-address \
  --tags=iap-ssh --service-account=anthropocene-gpu-fit@anthropocene-508912.iam.gserviceaccount.com \
  --scopes=https://www.googleapis.com/auth/devstorage.read_write \
  --disk=name=$N,boot=yes,auto-delete=no,device-name=${N}-boot \
  --shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring \
  --metadata=enable-oslogin=TRUE,block-project-ssh-keys=TRUE \
  --labels=environment=research,purpose=anthropocene-training,generation=l4-gpu,role=r5-fitting,managed-by=claude
```
Spot variant: replace `--provisioning-model=STANDARD` with `--provisioning-model=SPOT` (keep --instance-termination-action=STOP, --max-run-duration=20h, --maintenance-policy=TERMINATE).

In-place switch of an existing Spot VM (not used here): stop it, then `gcloud compute instances set-scheduling NAME --zone ... --provisioning-model=STANDARD --no-preemptible --maintenance-policy=TERMINATE --max-run-duration=20h --instance-termination-action=STOP`.

Another zone (-a or -b) if -c stays out of stock:
```bash
gcloud compute disks create anthropocene-gpu-fit-l4 --zone us-central1-a --source-snapshot anthropocene-gpu-fit-l4-boot-20261026 --type pd-balanced
```
then the create command above with `--zone us-central1-a`. The regional L4 quota covers all three zones. Disks are zonal, so use the snapshot copy rather than the original disk in another zone.

## Access and host key
- IAP SSH only: `gcloud compute ssh anthropocene-gpu-fit-l4 --zone us-central1-c --project anthropocene-508912 --tunnel-through-iap --strict-host-key-checking=yes --command="..."`.
- Pinned ED25519 host key: SHA256:goiAJ6avuCJNnUTHQw6EBV6JMmPhY3gHC/OuOF1gK2A.
  - It persists on the disk, so it is unchanged on a recreate from the disk or snapshot.
  - Verify with `ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub` from the first SSH session. Anything else: stop.
- The VM has no internet. Only GCS works, via Private Google Access.
- Wait about 60-90 s after start before the first SSH.

## What is on the disk (home: /home/suyashbhatt11_gmail_com)
- `~/r5gpu/venv`: torch 2.14.1+cu130 plus the r5 dependencies, installed offline from `~/r5gpu/wheelhouse`.
- `~/r5gpu/code/rl`: 51c139a (bootstrap and CUDA tests). `~/r5gpu/code-a204cf5/rl`: a204cf5 (24 device tests passed).
- `~/r5gpu/code-collect/rl`: 9da2c07 tarball contents with a local git commit 3960ba4e (tree e7d731b9... equals upstream 9da2c07). The collector needs a git commit, so this was created.
- `~/r5gpu/bridge/{dotnet,bridge}`: portable bridge (dll sha256 b5439c49...).
- `~/r5gpu/collect`: actor-normalizer.pt, tarballs, `out.preempted1`, `out.preempted2` and any later `out` (logs only).
- `~/r5gpu/PROVENANCE-learner.txt`, `~/launch.sh` (parity and heldout50 launcher), `~/diag.sh`, `~/t.sh`.
- Staged copies of the scripts are in gs://anthropocene-rl-datasets-508912/model9/r5/scripts/.

## Bootstrap workarounds already applied (do not rerun bootstrap)
- Ubuntu accelerator image has no `python3-venv`/`ensurepip` and no pip. The venv was made with `python3 -m venv --without-pip`, then pip was installed from `pip-26.2.2-py3-none-any.whl` (sha256 fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70, staged in the scripts/ GCS prefix).
- The r5 `inputs/` prefix did not exist when bootstrap ran, so bootstrap stopped at the input rsync. Smoke tests were run by hand.
- No C compiler on the VM, so Triton kernels cannot compile. `TORCH_DISABLE_NATIVE_JIT=1` must be exported BEFORE Python starts for every CUDA run (labor_device.configure refuses CUDA otherwise, in a204cf5).
- Collection is CPU-only (labor_device refuses a GPU for game-bound runs).

## Known risks
- Spot preemptions come about every 1-1.5 h. The collector writes shards only at the end, and there is no resume.
- Spot L4 capacity and on-demand L4 capacity in -c were both unreliable. On-demand returned STOCKOUT.
