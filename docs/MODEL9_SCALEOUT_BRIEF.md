# model9 scale-out brief: first runs on the 64-vCPU machine (20 October 2026)

This brief is self-contained for a fresh session on the scale-out machine. The history and evidence are in
`docs/MODEL9_M7.md`, starting at "Decisive stage-3 experiment".

## Where things stand

- The learned Village Elder (model9, contract 9) is trained by behaviour cloning from the DemandControl heuristic,
  then PPO with a teacher-KL anchor, a Lagrangian guard and per-regime advantage balancing.
- Commitments (capital spans) are implemented but **off**: train without `--commitments`, and evaluation honours
  that setting.
- The laptop experiment (5 seeds per arm, 104 worlds) found:
  - stage 3 (mixed use, free routes) against stage 2 continued is **inconclusive** on default worlds:
    0.987 [0.955, 1.017];
  - **neither arm improves default worlds over its init in 40 iterations.** Gains are in the capability fixtures
    (farming +7.7%);
  - seed-to-seed spread, not world count, sets the interval;
  - the gate's "no extra extinction" rule is decided by single events.

## Repositories and inputs

| What | Where |
| --- | --- |
| Simulator | `github.com/GeeseGaggle-211/Anthropocene`, branch `feature/elder-model9-native`, commit 78e4186 or later |
| RL | `github.com/GeeseGaggle-211/Anthropocene-RL`, branch `feature/elder-model9-native`, commit a5e6b74 or later |
| Inputs | `gs://anthropocene-rl-datasets-508912/model9/scaleout-inputs/` |
| Evidence so far | `gs://anthropocene-rl-datasets-508912/model9/ppo-c9/decisive/` |

The inputs folder holds:
- `stage1-seed2.pt`: the stage-1 gate init (SHA-256 prefix C2F424289029);
- `stage2-span-seed0-best.pt`: the shared stage-2 init (A647D2C4667C);
- `grassland-herding-seeds.txt` (training) and `grassland-heldout-seeds.txt` (evaluation).

## Setup (Linux)

1. Install the .NET 8 SDK and Python 3.11.
2. Clone both repositories side by side. On Windows clients, Git needs `core.longpaths=true`.
3. Build the bridge against the simulator checkout:
   ```
   dotnet build -c Release integration/Anthropocene.Model9Bridge.csproj -p:AnthropoceneSimRoot=<simulator checkout> -o integration/bin/model9
   ```
4. Create the venv: `python3.11 -m venv .venv && .venv/bin/pip install -e .`. Then
   `pytest -q tests/test_autonomous_v7_ppo.py tests/test_autonomous_v7.py` must pass (30 tests).
5. Set `ANTHROPOCENE_DOTNET` if `dotnet` is not on PATH.
6. Copy the inputs: `gcloud storage cp -r gs://.../model9/scaleout-inputs ./inputs`.
7. Smoke test, a few seconds: `python -m anthropocene_rl.autonomous_v7.ppo --init inputs/stage2-span-seed0-best.pt
   --stage capability --experts --seeds 1000-1003 --worlds 2 --turns 4 --iterations 1 --out runs/smoke`, run from
   `src/`.

## Resource model (measured on the 12-thread laptop)

Training:
- each run starts one bridge process per world (12) and uses 4 torch threads;
- it takes about 3 minutes per iteration with 3 runs sharing 12 threads, and about 3.5 GB RAM per run.

Evaluation:
- `stage_eval` runs about 1,500 rollouts per hour with 10 workers on an otherwise idle laptop.

On 64 vCPUs:
- start with **12 concurrent trainings** if RAM is at least 48 GB, check iteration time, then adjust;
- give evaluations `--workers 48` when nothing else runs;
- record the vCPU count, RAM and measured throughput in the results.

## Rules

- CONFIRMATION seeds 3000-3059 stay untouched.
- Training uses TRAIN 1000-1599. Selection uses 2020-2025. Gates use DEVELOPMENT 2000-2019 and 2026-2059, plus
  the held-out grassland seeds.
- **Pre-register** each experiment's primary measure and decision rule in `docs/MODEL9_M7.md`, and commit it before
  launching.
- Never print credentials or tokens.
- Commit code and docs to the RL branch. Upload evidence to `gs://.../model9/ppo-c9/<experiment>/`.

## The VM (revised 20 October)

- `anthropocene-trainer-60`, project `anthropocene-508912`, zone `us-central1-c`: Spot c3d-highcpu-60 (60 vCPU,
  115 GiB), Ubuntu 24.04.
- It auto-stops after 8 hours. Spot preemption also stops it, and the local disk survives.
- SSH through IAP only, with `--strict-host-key-checking=yes`. The pinned key is
  SHA256:qv55HFWP7NUJyGC4CvB6H8vvFO8YeNANHgAeYVi0zfE; stop on any host-key prompt.
- **Only one agent controls it.** Check its status before starting, and stop it when done.
- Environment in `~/m9-bench`:
  - venv at `venv`, code at `rl/src` (replace it with the current `src`; the bridge has not changed since 20fed92);
  - bridge at `rl/integration/bin/model9`.
- Measured throughput: 18 arms is about 2.1-2.2M decisions per hour. Run 18-24 arms with `OMP_NUM_THREADS=3`,
  detached with `setsid nohup`.
- No resume and no GCS upload: results come back by `gcloud compute scp`, then go to GCS from the laptop.
- Each batch must fit in the 8-hour cap, or be restartable.

## Experiments, in order (revised 20 October)

The user decided to train stage 3 and learned movement together, as a 2x2 of movement (native or learned) against
stage (capability or mixed). That waits for contract-9 movement and a cloned movement head within about 2-3% of the
teacher. Until then the VM runs E2 (pre-registered in `docs/MODEL9_M7.md`: learning rate, teacher anchor and batch
size, 24 arms, equal decision budgets) and E3.

The economy-only ceiling (`--model9-oracle` on default worlds, simulator side) runs on the laptop.

The original E1 below is kept for reference only; it is superseded by the 2x2.

### E1 (superseded): the stage-3 question, at power

Same design as the laptop's decisive experiment (`model9-data/stage3-decisive.ps1`, in the evidence folder), with
20 seeds per arm:
- **Treatment:** `--stage mixed`. **Control:** `--stage capability`. Seeds 10-29 for both.
- Shared settings: init `stage2-span-seed0-best.pt`; `--experts --seeds 1000-1599 --grassland-seeds
  grassland-herding-seeds.txt --worlds 12 --turns 25 --iterations 40 --eval-every 10 --eval-seeds 2020-2025
  --lagrange-lr 2`.
- Evaluation: `stage_eval --init stage1-seed2.pt --default-seeds 2000-2019,2026-2059 --fixture-seeds 2000-2019
  --grassland-seeds grassland-heldout-seeds.txt --grassland-count 10`, on `latest.pt`.
- Comparison: `paired_compare --a <control> --b <treatment>`.
- Decision rule: the interval should come within about ±1.5%. Below 1 means stage 3 regresses; within ±2% means
  mixed-use training is safe to carry forward.
- Cost: 40 runs of 40 iterations, about 4 hours with 12 concurrent, plus about 2-3 hours of evaluation.

### E2: why default worlds plateau

The control showed no default-world gain in 40 iterations. Before longer runs are worth anything, find out which
lever moves default worlds. Stage 2, 5 seeds per cell, 100 iterations:
- baseline;
- 4x the batch (`--worlds 48`);
- a weaker teacher anchor;
- a higher learning rate (`--lr`, 2x and 0.5x the default).

The anchor weight is a `Hyper` field with no flag yet: add one first, with a test. Measure the per-cell default-world
ratio against the init, with intervals. Lagrangian costs and the welfare/QoL breakdown should be reported alongside,
because welfare is an interim reward and a gain bought by gaming it must be flagged.

### E3: a robust gate

Replace "no more extinctions than the init" with an extinction **rate** and an interval, pooled over seeds. Compare
both rules on E1's evaluations before adopting.

## Report back

Report E1 (and E2 if run) with:
- the pre-registered verdict;
- the per-regime table with intervals;
- per-seed spread;
- measured throughput;
- anything that contradicts this brief.
