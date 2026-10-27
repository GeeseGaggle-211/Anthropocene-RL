# r5 GPU fitting VM: plan for review

Status: **prepared, nothing created.** No gcloud command that creates, starts or
modifies a resource has been run. The co-architect reviews this plan first, and the
user must request GPU quota before anything can be created (see Quota).

Scope: only the r5 stages that train on stored shards and never run the game:

- the critic acceptance fits, 2 arms x 5 seeds (`labor_r5_acceptance run`, which runs
  pretrain-critic, the A-head fit and evaluate for each arm x seed, plus the determinism
  probe);
- actor BC (`labor_training pretrain-bc --dataset ...`).

Collection, search, DAgger collection, the T4 trajectory recorder and every evaluation that
plays the game stay on `anthropocene-trainer-60` (CPU).

Learner support: branch `feature/r5-integration`, commit `9da2c07` (pushed). The change
adds `--device {cpu,cuda,auto}` to pretrain-critic, pretrain-bc, `labor_advantage fit`, and
`labor_r5_acceptance evaluate|run`. `run` accepts only `cpu|cuda`, so the receipt records
an explicit device. See "Learner changes" below.

## 1. GPU choice and quota

| Item | Finding (read-only, 26 Oct 2026) |
| --- | --- |
| L4 zones in us-central1 | `us-central1-a`, `-b`, `-c` (`nvidia-l4` and `-vws`). Default: `us-central1-c`, the same zone as the CPU VM. |
| G2 shapes in us-central1-c | g2-standard-4/8/12/16/32 (1 L4), -24 (2), -48 (4), -96 (8) |
| **GPUS_ALL_REGIONS (project)** | **limit 0, usage 0. Blocking: no GPU VM can be created until the user requests at least 1.** |
| NVIDIA_L4_GPUS (us-central1) | limit 1, usage 0 |
| PREEMPTIBLE_NVIDIA_L4_GPUS (us-central1) | limit 1, usage 0 (Spot uses this quota) |
| **CPUS_ALL_REGIONS (project)** | **limit 60, usage 60.** The running c3d-highcpu-60 uses all of it. A g2-standard-16 needs 16 more vCPUs, so request **80**, or run the GPU VM only while the CPU VM is stopped. |
| CPUS (us-central1, regional) | limit 200, usage 0 |
| DISKS_TOTAL_GB / SSD_TOTAL_GB (us-central1) | 0/4096 and 100/500: a 200 GB pd-balanced disk fits |

**Quota requests for the user** (Console > IAM & Admin > Quotas & System Limits, or the
commands below, which the user runs; they are not part of this plan's automation):

```bash
gcloud beta quotas preferences create --project=anthropocene-508912 \
  --service=compute.googleapis.com --quota-id=GPUS-ALL-REGIONS-per-project \
  --preferred-value=1 --preference-id=r5-gpu-all-regions --justification="r5 critic fits on one Spot L4"
gcloud beta quotas preferences create --project=anthropocene-508912 \
  --service=compute.googleapis.com --quota-id=CPUS-ALL-REGIONS-per-project \
  --preferred-value=80 --preference-id=r5-cpus-all-regions --justification="Spot g2-standard-16 beside the c3d-60"
```

Machine: **g2-standard-16** (1 L4 24 GB, 16 vCPU, 64 GB RAM). Host RAM is the limit, not the
GPU. A critic fit holds about 11 GB of dataset in host memory (plus about 2 GB of process
overhead), so 64 GB runs **4 concurrent fits**. A g2-standard-12 (48 GB) runs 3, and a
g2-standard-8 (32 GB) runs only 2. The data stays in host memory by design: only modules and
minibatches go to the GPU.

## 2. Cost (estimates, USD; verify against the billing SKU list)

| Item | Rate (estimate) | Use | Cost |
| --- | --- | --- | --- |
| g2-standard-16 Spot | $0.50 to 0.85/h (third-party g2-standard-8 Spot snapshots are $0.40 to 0.71/h; on-demand -8 is $0.854/h) | 6 to 14 h (sec. 3, with 1.5x preemption slack) | $3 to 12 |
| 200 GB pd-balanced boot disk | ~$0.10/GB-month, about $0.66/day, billed while the VM exists, even when stopped | 7 days, then delete | ~$5 |
| GCS egress, bucket `asia-northeast1` to `us-central1` (intercontinental) | ~$0.08 to 0.12/GiB | 30 to 60 GiB of inputs in; under 5 GiB of outputs back | $3 to 8 |
| Wheelhouse in GCS (sec. 5), replaces Cloud NAT | ~$0.02/GiB-month storage plus cross-region egress ~$0.08 to 0.12/GiB (bucket is asia-northeast1) | ~6 GiB of wheels | < $1 |
| **GPU side, worst case** | | | **~$25** |
| Existing CPU VM c3d-highcpu-60 Spot (continued) | ~$0.75 to 1.10/h (estimate), plus 100 GB disk ~$0.33/day and its external IP ~$0.005/h | all game-bound r5 stages | the rest of the budget |

Against the **$260** credit: the GPU side costs at most about $25, which leaves **≥ $235,
roughly 210 to 310 CPU-VM hours** at the estimated Spot rate. Moving the critic fits off the
CPU VM also frees about 15 to 20 CPU-VM hours (4 fits x 14 threads), worth about $12 to 20.
So the GPU is roughly cost-neutral; what it buys is wall time and a free CPU VM for
collection and search. Inputs total `gcloud storage du -s <prefix>` once staged; the 30 to
60 GiB figure is an assumption.

## 3. Time estimates and assumptions

Workload (from the brief): a critic fit is about 400 steps/epoch x 4 epochs = **1,600 steps**,
with 4 TD refreshes per epoch (a forward pass over every row, about every 100 steps). On the
CPU VM it takes about 5.8 s/step at 28 threads, so 10 fits = 16,000 steps = **26 fit-hours**.
BC takes about 2 s/step on CPU.

Why the speedup is uncertain: each encoder is about 0.8M parameters over 121 tokens, and the
task critic plus two cost critics make three encoders. Per step that is roughly
6 x 0.8M x 121 x 256 ≈ 0.15 TFLOP per encoder, about 0.5 TFLOP in all including the amortized
TD refresh. That is consistent with the measured ~80 GFLOP/s on CPU. On an L4 (fp32, TF32
deliberately off for determinism, ~30 TFLOP/s peak, 25 to 40% realistic for these small
matmuls), compute is about 0.05 s/step. The kernels are small, so launch and Python overhead
dominate (three transformer encoders forward and backward, plus the stability monitor's
per-step snapshot, now one host sync for the finite checks): about 0.1 to 0.3 s/step. TD
refreshes also pay host-side minibatch gathering and the numpy privileged normalizer. GPU
utilisation will be modest for one fit, which is why fits run concurrently.

| Assumption | Optimistic | Central | Pessimistic |
| --- | --- | --- | --- |
| One fit alone on the L4, s/step | 0.15 | 0.35 | 1.0 (only ~6x faster than CPU) |
| 4 concurrent fits, slowdown per fit | 1.5x | 2x | 2x |
| One critic fit, 4 concurrent (1,600 steps) | 6 min | 19 min | 53 min |
| 10 critic fits (waves of 4, 4, 2) | 0.3 h | 1.0 h | 2.7 h |
| Dataset load per fit (11 GB torch.load from local disk) | 2 min | 4 min | 8 min |
| A fit per (arm, seed), GPU (assumed small head; not timed on CPU in the brief) | 3 min | 8 min | 15 min |
| evaluate per fit (forward only on calibration and reserved data) | 3 min | 8 min | 15 min |
| determinism probe (2 x 50 critic steps + 2 short A fits) | 3 min | 5 min | 10 min |
| **Whole acceptance run, JOBS=4** | **~1 h** | **~2 h** | **~4.5 h** |

BC (CPU ~2 s/step; GPU 0.1 to 0.5 s/step, one actor encoder): the step count depends on the
BC plan, so the table scales by steps.

| BC steps | CPU | GPU |
| --- | --- | --- |
| 1,600 (e.g. 2 epochs at ~800 steps) | 0.9 h | 3 to 13 min |
| 10,000 | 5.6 h | 0.3 to 1.4 h |
| 20,000 | 11 h | 0.6 to 2.8 h |

GPU hours budgeted: bootstrap and smoke 1 h, acceptance run 1 to 4.5 h, BC 0.5 to 3 h, all
times 1.5 for preemption and reruns, giving **6 to 14 h**. Measure the central assumption in
the first 10 minutes: the `[progress] steps=... elapsed_s=...` lines in `fit.log` give s/step.

**Memory per fit (estimate):** host about 13 GB. GPU about 1.5 to 4 GB: parameters, grads and
Adam state for ~2.5M parameters are about 40 MB; the monitor snapshot is about 30 MB;
activations for batch 256 x 121 tokens x 3 encoders are about 1 to 3 GB; the CUDA context is
about 0.5 GB. Four fits use about 16 GB of the 24 GB. Each non-CPU fit now records
`device.max_memory_allocated_gib` in its summary (checkpoint and A artifact), so the first
wave measures this.

## 4. Learner changes (Part 1, committed)

New module `src/anthropocene_rl/autonomous_v7/labor_device.py`. In outline:

- `--device cpu` is the default and the historical path, bit for bit. Every move is a no-op
  on CPU, no global flag changes on CPU, and `labor_r5_acceptance run` adds no `--device`
  to CPU sub-commands, so CPU command lines are unchanged.
- Modules move to the device once, and each minibatch or chunk moves per step. The datasets,
  RNGs (numpy shuffles), TD targets and saturation statistics stay on the host. Model
  initialization (`initialize`, the A head under `torch.manual_seed`) stays on CPU, so
  initial parameters are identical on every device.
- **Checkpoints are device-agnostic:** the commands move the modules back to CPU before
  saving. The critic resume state, the optimizer state and the A artifact are saved through
  `cpu_copy`. A CUDA resume state resumes on CPU and the reverse (tested).
- **Stability monitor:** rollback restores parameters and Adam state on the device, since
  `load_state_dict` follows each parameter's device. The finite checks are one host sync per
  device (`all_finite`). Adam's CPU `step` counters keep the historical per-tensor loop.
  The attention-mass and dead-feature probes are device-safe.
- **Determinism:** CPU keeps the approved bitwise and 1e-6 guarantee. CUDA sets cuDNN
  deterministic, benchmark off, TF32 off, `CUBLAS_WORKSPACE_CONFIG=:4096:8` and
  `use_deterministic_algorithms(True, warn_only=True)`. The acceptance harness's determinism
  check is device-aware: CPU uses the approved 1e-6 relative, and CUDA uses
  **`determinism_tolerance_cuda` = 1e-5 relative, status approved (user 26 Oct)**. A
  non-bitwise CUDA probe within 1e-5 is **PASS**; beyond it is FAIL; bitwise is PASS.
- **Game-bound refusal:** `--worlds` (in-process collection) with a non-CPU device is
  refused, so only `--dataset` fits use the GPU. The T4 trajectory recorder and `initialize`
  stay on CPU inside `run`.
- **Tests:** `tests/test_labor_device.py`. Its CPU tests always run. Its CPU-vs-CUDA parity
  tests are skipped without a GPU:
  - one-step loss and gradient agreement within 1e-4 relative for the critics and BC;
  - a teleported mid-trajectory step (the CPU trajectory's state loaded on both devices)
    for the critic, the A head and BC: loss and global-L2 gradient within 1e-4;
  - short monitored critic, A and BC fits within 1e-3 of CPU **in output space**
    (worst per-output change-relative drift); per-tensor parameter drift is printed, not
    gated, because gauge parameters make it ill-posed (feature/r5-cuda-diag,
    `diag-proposal.md`);
  - two CUDA critic fits within the approved 1e-5 (bitwise on the L4; the remedy if it ever
    fails is pinning `SDPBackend.MATH`, never loosening);
  - a CUDA-written resume state continued on CPU.

  On the GPU VM, `bootstrap.sh` requires them to run, not skip.
- **`TORCH_DISABLE_NATIVE_JIT=1` for every CUDA process.** torch 2.14 registers
  `torch._native` Triton overrides (an `aten::bmm` outer-product kernel that the actor's
  einsum backward hits) that compile with a C compiler at first use; the VM has none and no
  internet. torch reads the variable once, during `import torch`, so it must be in the
  environment before Python starts. `bootstrap.sh` exports it, `tools/r5_critic_acceptance.sh`
  and `tools/g0launch.py` set it, and `labor_device.configure` refuses CUDA when native JIT
  kernels are registered. Any other CUDA command (an ad hoc `labor_training ... --device
  cuda`, `pytest`) needs `export TORCH_DISABLE_NATIVE_JIT=1` first.

**Verified on the L4 (26 Oct, torch 2.14.1+cu130, driver 580):** one-step parity, cross-device
resume and bitwise CUDA repeats; the multi-step per-tensor checks failed on gauge parameters
and were replaced by the output-space check above (feature/r5-cuda-diag).

## 5. Network, identity and image

- **No external IP, IAP SSH:** same network, subnet and `iap-ssh` tag as the CPU VM
  (`anthropocene-training`, `anthropocene-training-us-central1`). Note: the CPU VM currently
  *has* an external IP (`external-nat`). The GPU VM has none.
- **Network decision (approved, option a): no external IP + Private Google Access + an
  offline package cache in GCS. No Cloud NAT.** The VM reaches only Google APIs (GCS for the
  code tarball, r5 inputs and the wheelhouse; IAP for SSH). It cannot reach PyPI,
  `download.pytorch.org` or apt, and `bootstrap.sh` has no online fallback.
- **Private Google Access check (read-only, 26 Oct 2026):**

  ```bash
  gcloud compute networks subnets describe default --region=us-central1     --project=anthropocene-508912 --format="value(privateIpGoogleAccess)"
  ```

  Result: **False** for subnet `default` (network `default`). However, `create-gpu-vm.sh`
  does not use `default`: it uses subnet `anthropocene-training-us-central1` on network
  `anthropocene-training`, which reports **privateIpGoogleAccess = True**
  (`gcloud compute networks subnets list` confirms). So the planned VM already has Private
  Google Access and **no enabling step is needed**. Only if the VM is ever placed on the
  `default` subnet instead, the co-architect must first run (not run by this plan):

  ```bash
  gcloud compute networks subnets update default --region=us-central1 --enable-private-ip-google-access
  ```
- **Offline wheelhouse.** On the CPU VM (internet access), run `build-wheelhouse.sh`. It
  downloads CUDA torch 2.14.1 (cu128, a cu12x build that the L4 image's NVIDIA 580 driver
  runs) with all of its dependencies, the repo's pinned runtime dependencies and pytest, for
  CPython 3.12 on Ubuntu 24.04 x86_64 (manylinux, binary wheels only). It proves the set is
  complete by installing it with `--no-index` into a scratch venv, writes `requirements.txt`
  and a `MANIFEST.sha256` (sha256 of every file), and uploads to
  `gs://anthropocene-rl-datasets-508912/model9/r5/wheelhouse/`:

  ```bash
  bash build-wheelhouse.sh              # on anthropocene-trainer-60
  UPLOAD=0 bash build-wheelhouse.sh     # build and verify only
  ```

  `bootstrap.sh` rsyncs that prefix, checks every wheel against the manifest, then runs
  `pip install --no-index --find-links <wheelhouse> torch==2.14.1 -r requirements.txt`. A
  missing wheel is a hard error. The image's Python must match `PY_VERSION` (3.12 on Ubuntu
  24.04); confirm with `python3 --version` on the VM, and rebuild with `PY_VERSION=` if not.
  If torch 2.14.1 has no `cu128` wheel, rerun with `CUDA_TAG=cu126` or `cu130`.
  Read-only preflight for the rest of the network:

  ```bash
  gcloud compute firewall-rules list --project anthropocene-508912     --filter="network:anthropocene-training" --format="table(name,sourceRanges.list(),targetTags.list(),allowed[].map().firewall_rule().list())"
  ```
- **Service account** `anthropocene-gpu-fit@anthropocene-508912.iam.gserviceaccount.com`:
  no project roles. It has `roles/storage.objectUser` on `gs://anthropocene-rl-datasets-508912`
  only, and the VM's OAuth scope is `devstorage.read_write`. It is new, so the CPU VM's
  `anthropocene-trainer` account (cloud-platform scope) is not reused.
- **Image:** `ubuntu-os-accelerator-images/ubuntu-accelerator-2404-amd64-with-nvidia-580`,
  Ubuntu 24.04 with a preinstalled signed NVIDIA 580 driver (CUDA 13 capable) and Python
  3.12. Shielded VM and Secure Boot are on, as on the CPU VM. The Deep Learning VM PyTorch
  images ship torch 2.9 (`pytorch-2-9-cu129-*`), not the CPU VM's **torch 2.14.1**
  (pinned in `pyproject.toml`), so they are not used. `bootstrap.sh` installs CUDA torch
  2.14.1 into a venv from the offline wheelhouse (cu128 by default). The CPU VM's exact torch
  build should be confirmed (`python -c 'import torch; print(torch.__version__)'` in
  `~/m9-bench/venv`). CPU and CUDA builds of the same version share the Python code; only
  the kernels differ, which the parity tolerances cover.
- **Spot:** termination action STOP, max run duration 20 h, no automatic restart, no
  start-schedule policy (the CPU VM's `g0-autostart4` is deliberately not copied).

## 6. Risks

- **Quota:** with GPUS_ALL_REGIONS = 0, creation fails. With CPUS_ALL_REGIONS at 60/60,
  creation also fails while the CPU VM runs, unless the request for 80 is granted.
- **Spot preemption** (STOP keeps the disk). The critic fit writes an exact resume state
  every `RESUME_EVERY` steps (harness default 100; use 100 on the GPU, about 1 minute of
  work). A rerun of the same `run` command continues bit for bit and skips finished stages.
  BC through `tools/g0launch.py` writes `<out>.partial` every 250 steps (`G0_SAVE_EVERY`),
  and Adam moments restart on resume. The A fit and evaluate have no partials; they are
  minutes long and rerun from scratch. Spot L4 capacity in us-central1-c can be
  unavailable; zones a and b are fallbacks (set `ZONE=`), but they are not the CPU VM's zone.
- **20 h max-run-duration** stops the VM mid-run if a session overruns; resume as for a
  preemption.
- **Data transfer:** inputs cross continents from the asia-northeast1 bucket (~$0.08 to
  0.12/GiB, 30 to 60 GiB assumed). `gcloud storage rsync` is incremental, so reruns are cheap.
  The alternative is a direct same-zone copy from the CPU VM over the internal network,
  which is free but needs VM-to-VM SSH; not planned.
- **Driver and Secure Boot:** the accelerator image's driver is signed. If `nvidia-smi`
  fails, `bootstrap.sh` stops. The fallback is to recreate with `--no-shielded-secure-boot`,
  which needs review.
- **Determinism tolerance:** `determinism_tolerance_cuda` = 1e-5 is approved (user
  26 Oct); a CUDA probe within it PASSes. Beyond it FAILs.
- **Wheelhouse drift:** the wheelhouse is a snapshot. Rebuild it (and re-verify the
  manifest) if the torch pin, `pyproject.toml` dependencies or the image's Python change.
- **GPU-fitted results are not bitwise equal to CPU-fitted ones.** Arms and seeds must all
  be fitted on the same device for the arm comparison. Do not mix CPU and GPU fits in one
  receipt; the receipt records `configuration.device`.
- **Host RAM:** 4 critic fits plus a concurrent evaluate (which loads calibration data)
  can approach 64 GB. If `dmesg` shows OOM kills, use JOBS=3.

## 7. Exact commands (for review; none run)

Read-only preflight: see section 5, plus the quota check printed by `create-gpu-vm.sh`.

**Stage inputs from the CPU VM** (on anthropocene-trainer-60; paths are the CPU VM's r5
inputs, as used by `tools/r5_critic_acceptance.sh`):

```bash
P=gs://anthropocene-rl-datasets-508912/model9/r5/inputs
gcloud storage cp "$NORMALIZER" $P/normalizer.pt
gcloud storage cp "$PRIV_NORMALIZER" $P/privileged-normalizer.pt
gcloud storage cp "$CONTROL_REPORT" $P/control-report.json
gcloud storage rsync -r "$FIT_DATA" $P/fit-data/$(basename "$FIT_DATA")   # each FIT_DATA dir
gcloud storage rsync -r "$CAL_DATA" $P/cal-data
gcloud storage rsync -r "$FIT_PAYLOADS" $P/fit-payloads                 # each dir
gcloud storage rsync -r "$RESERVED_PAYLOADS" $P/reserved-payloads       # each dir
gcloud storage rsync -r <bc teacher/aggregated dataset dirs> $P/bc-data/<name>
gcloud storage cp <BC start checkpoint> $P/checkpoints/
gcloud storage du -s $P
```

**Code tarball** (workstation, from the r5 worktree, after the Part 1 commit is pushed):

```bash
cd C:/Users/suyas/Documents/ChatGPT/worktrees/r5-int-rl
git fetch && git archive --format=tar.gz --prefix=rl/ -o r5-int-rl-<sha>.tar.gz <sha>
sha256sum r5-int-rl-<sha>.tar.gz
gcloud storage cp r5-int-rl-<sha>.tar.gz gs://anthropocene-rl-datasets-508912/model9/r5/code/
```

**Build the wheelhouse** (once, on anthropocene-trainer-60, before bootstrap; needs internet):

```bash
gcloud compute scp build-wheelhouse.sh anthropocene-trainer-60:~ --zone us-central1-c --tunnel-through-iap
# on the CPU VM:
bash ~/build-wheelhouse.sh
```

**Create** (after approval and quota; dry-run first):

```bash
./create-gpu-vm.sh                 # prints the commands only
DRY_RUN=0 ./create-gpu-vm.sh       # creates SA, bucket binding, VM
```

**Bootstrap** (on the GPU VM):

```bash
gcloud compute scp bootstrap.sh anthropocene-gpu-fit-l4:~ --zone us-central1-c --tunnel-through-iap
gcloud compute ssh anthropocene-gpu-fit-l4 --zone us-central1-c --project anthropocene-508912 --tunnel-through-iap
CODE_TARBALL=gs://anthropocene-rl-datasets-508912/model9/r5/code/r5-int-rl-<sha>.tar.gz \
CODE_SHA256=<sha256> bash ~/bootstrap.sh        # installs from the GCS wheelhouse only
```

**Critic acceptance run** (on the GPU VM, inside `tmux`, so the shell survives disconnects):

```bash
R=~/r5gpu; I=$R/inputs
cd $R/code/rl
PYTHON=$R/venv/bin/python DEVICE=cuda JOBS=4 THREADS=3 RESUME_EVERY=100 \
NORMALIZER=$I/normalizer.pt PRIV_NORMALIZER=$I/privileged-normalizer.pt \
FIT_DATA="$(ls -d $I/fit-data/* | tr '\n' ' ')" CAL_DATA=$I/cal-data \
FIT_PAYLOADS="$I/fit-payloads" RESERVED_PAYLOADS="$I/reserved-payloads" \
CONTROL_REPORT=$I/control-report.json OUT=$R/out/r5-acceptance-gpu ATTEMPT_ID=r5-gpu-<n> \
bash tools/r5_critic_acceptance.sh
# after a preemption: start the VM, ssh, rerun the identical command.
```

**Actor BC** (on the GPU VM; partials every 250 steps):

```bash
cd ~/r5gpu/code/rl && PYTHONPATH=src G0_SAVE_EVERY=250 ~/r5gpu/venv/bin/python tools/g0launch.py \
  pretrain-bc --checkpoint ~/r5gpu/inputs/checkpoints/<start>.pt \
  --dataset ~/r5gpu/inputs/bc-data/<dir> [--dataset ...] \
  --epochs <E> --batch-size 256 --lr <lr> --device cuda --out ~/r5gpu/out/bc/<name>.pt
```

**Return results and stop:**

```bash
gcloud storage rsync -r ~/r5gpu/out gs://anthropocene-rl-datasets-508912/model9/r5/gpu-out/<attempt>
sudo shutdown -h now                      # or, from the workstation:
gcloud compute instances stop anthropocene-gpu-fit-l4 --zone us-central1-c --project anthropocene-508912
# when the round's GPU work is done (stops disk billing):
gcloud compute instances delete anthropocene-gpu-fit-l4 --zone us-central1-c --project anthropocene-508912
```

## Files

- `create-gpu-vm.sh`: service account, bucket-only IAM and Spot VM. It defaults to a dry
  run, which only prints the commands.
- `bootstrap.sh`: run on the VM. It checks the GPU, fetches the code tarball, builds a CUDA
  torch 2.14.1 venv (wheelhouse only), rsyncs the r5 inputs, then runs the smoke and
  parity tests.
- `build-wheelhouse.sh`: run on the CPU VM. It builds, verifies and uploads the offline
  wheelhouse (CUDA torch 2.14.1, dependencies, sha256 manifest) to GCS.
- `README.md`: this plan.
