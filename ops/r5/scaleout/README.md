# r5 scale-out: critic V (GPU), search with full payloads (CPU), actor BC pilot

Branch `ops/r5-scaleout` (from `feature/r5-integration` 407958d). Nothing here has been run on
a VM. Operators run it; every stage script is resumable, uploads its outputs to
`gs://anthropocene-rl-datasets-508912/model9/r5/scaleout/` and ends with
`sudo shutdown -h now` from an EXIT trap, on success and on failure (`NO_SHUTDOWN=1` keeps the
VM up for debugging).

| File | Runs on | What |
|---|---|---|
| `gpu_queue.sh` | GPU VM | inputs → QoL reference scale on all 8 fit batches → actor BC pilot (`BC=1`) → critic V scale-out (`STAGE=critic`) → upload → shutdown |
| `actor_bc_pilot.sh` | GPU VM (called by `gpu_queue.sh`), or CPU VM | initialize → BC on fit8-b01..b04 (brief schedule) → same-state fidelity → upload |
| `cpu_queue.sh` + `cpu_sched.py` | CPU VM | search chunks (reserved + fitting H12 roots, full payloads) and the actor panel (24 dev worlds × 50 decisions) under one slot/memory budget → upload per task → shutdown |
| `panel_report.py` | CPU VM (called by the scheduler) | merges the per-world panel reports, applies the manifest §8 actor gate vs H12 and reports brief S1 vs H |
| `lib.sh` | both | logging, GCS helpers, memory sampler, shutdown |

## 1. Harness changes on this branch (review these)

| Commit | Change | Tests |
|---|---|---|
| eb93ff3 | `labor_r5_acceptance run --stage critic` (shell: `STAGE=critic`): V fits with the seed gate, early stop and per-epoch validation, the **critic-only** determinism probe, and the V evaluation (relative ordering, TD(0), calibration) **without** search payloads. Writes `<arm>/seed-<s>/evidence-critic.json` and `critic-receipt.json` with a non-final `critic_stage` V-only verdict; A-dependent checks HOLD and are listed as deferred, never PASS. A later `--stage all` in the same `OUT` continues at the A fit (critics not refitted). `--gap-fit-data` (optional subset of fit data for the generalisation gap). Shell driver: `STAGE`, `GAP_FIT_DATA`, `LR`, `INIT_ARGS`, `CRITIC_ARGS`, `ADVANTAGE_ARGS`. `--stage all` (default) unchanged. | `tests/test_labor_r5_critic_stage.py` (9: input validation, run_fit command sequence and continuation, critic-only determinism, receipt and summary, seed gate on critic evidence, shell driver) |
| eb93ff3 | `labor_search --plan-fraction` (T2 plans on a keyed share of roots; approved 25%). Default 1.0 = every root; the source record is unchanged at 1.0. | 3 new in `tests/test_labor_search_plans_resume.py` |
| eb93ff3 | `pretrain-bc --lr-schedule warmup-cosine` (brief §5.1: `--lr` is the peak, 10% warm-up, cosine to 10%), `--lr-schedule-total-steps`, `--lr-schedule-start-step` (resumed fits and later rounds continue the program). `constant` (default) is the historical fit. | `tests/test_labor_bc_lr_schedule.py` (6) |
| 839d197 | `run/receipt --population-tail {advisory,gate}`, default advisory (Q1, user 28 Oct) | 3 in `tests/test_labor_r5_critic_stage.py` |
| feature/r5-gradnorm | `pretrain-critic --loss-balancing gradnorm` (`--gradnorm-alpha` 1.5, `--gradnorm-lr` 0.025; `--qol-loss-weight` is the initial ratio) and `--grad-clip per-encoder`, passed through `CRITIC_ARGS` / `run --critic-args`. Under gradnorm `gradient_dominance` is advisory and the mandatory `task_progress_stall` gates (early stop, seed gate; PREREGISTERED `stability_task_progress_stall`, user 29 Oct). Both off by default (bitwise unchanged). See `docs/LABOR_ADVANTAGE.md`. | `tests/test_labor_gradnorm.py` (15) |
| 19832cb | Other-head regression against the run's own seed-matched control arm when no control report is given (Q2, user 28 Oct); `regression_reference` in the receipt | 2 in `tests/test_labor_r5_critic_stage.py` |
| 71782ee | `labor_search` root-level resume (Q4, user 28 Oct): a checkpoint at every main-path decision boundary (snapshot of the previous boundary, transient state, executed payloads, successor digest, loop state, appended rows/tables/payload roots); `--resume` replays one decision, verifies the digest and continues, so a stop loses at most the in-flight decision and its root. Mismatch: the world is redone from its reset (`search_resume_fallback` in meta). `--checkpoint-keep` (2). | `tests/test_labor_search_resume.py` (9, fake world through the real loop), `tests/test_labor_search_resume_live.py` (bitwise SIGKILL gate on the bridge; skipped locally, run by `cpu_queue.sh` on the VM before search) |
| bd800d0 | Dataset loaders read shard arrays copy-on-write memory-mapped before `torch.cat`, so loading N GB peaks at ~N GB of anonymous memory instead of ~2N. Values unchanged. | 1 new in `tests/test_labor_multi_dataset.py`; multi-dataset, acceptance, gate, early-stop, BC, scenario, tail, resume suites pass locally (bridge-dependent live tests skipped/erroring as before: no local bridge) |

Not changed: thresholds, gates, the receipt's own verdict algebra, search semantics at default
flags, the critic or BC objectives.

## 2. Inputs

All in `gs://anthropocene-rl-datasets-508912/model9/r5/` (= `$GCS`):

| Input | Path | Notes |
|---|---|---|
| Fit batches (60 worlds, 252,261 rows, 26.3 GB) | `pilot/fit8-b01 b02 b03r b04 b05 b06 b07 b08` | b03r replaces b03 |
| Calibration (20 worlds, 91,384 rows, 9.5 GB) | `pilot/cal8-b01 b02 b03` | V validation and evaluation only |
| Actor normalizer | `pilot/actor-normalizer.pt` | sha256 3c8c5ca3… (checked by the scripts) |
| Privileged normalizer v2 | `post-collection/privileged-normalizer.pt` | fitted on fit8-b01..b08 |
| QoL reference scale | recomputed by `gpu_queue.sh` on all 8 fit batches → `scaleout/r5-scale-fit8.json` | replaces `pilot/out-v2/r5-scale.json` (5 batches) |

The CPU VM already has the same data in `~/r5/` (`actor-normalizer.pt`, batches). Search
reads only the actor normalizer; it plays the game.

## 3. Order and parallelism

Both VMs run at the same time (CPU quota 60 + 16 = 76 of 80).

```
t=0      GPU: stage inputs (~10 min) -> QoL scale (~10 min) -> BC pilot (~1.5-2.5 h)
         CPU: search starts at once (reserved chunks first, then fitting passes)
t~2.5h   GPU: actor uploaded -> critic V: candidate seed 0 (+ critic determinism), control seed 0
         CPU: scheduler fetches the actor; panel worlds take priority over fitting chunks (~4-5 h)
t~6.5h   GPU: seed gate (seed-gate.json). Passed -> seeds 1-4 x 2 arms; tripped -> receipt, shutdown
t~22h    GPU: critic-receipt.json, upload, shutdown (one restart: the 20 h max-run, see 7)
t~45h    CPU: PHASE1_DONE (reserved 32/world + 8 roots/fitting world): first A fit possible
t~70-95h CPU: SEARCH_H12_DONE, upload, shutdown (+ restart gaps after guardrail stops/preemptions)
```

The BC pilot runs on the GPU **before** the critic because the two cannot share 62 GB of host
RAM, and on the CPU it would take ~8-11 h at ~2.2 s/step (G4 refit timing) while competing with
search. It delays the critic receipt by ~2 h. To keep the critic first: `BC=0` on the GPU and
run `actor_bc_pilot.sh` on the CPU VM (`DEVICE=cpu`, section 4.3).

## 4. Exact commands

### 4.0 Code staging (once, on a machine with this repo)

```bash
SHA=$(git rev-parse --short=7 origin/ops/r5-scaleout)
git archive --format=tar.gz --prefix=rl/ -o r5-int-rl-$SHA.tar.gz $SHA
sha256sum r5-int-rl-$SHA.tar.gz > r5-int-rl-$SHA.tar.gz.sha256
gcloud storage cp r5-int-rl-$SHA.tar.gz r5-int-rl-$SHA.tar.gz.sha256 gs://anthropocene-rl-datasets-508912/model9/r5/code/
```

### 4.1 GPU VM (anthropocene-gpu-fit-l4, us-central1-c, on-demand)

Start it, verify the pinned host key (ED25519 SHA256:77KjMK4quoPVB53ABMd33WyLC7TsDhFR42B1b5jkR+c)
before the first SSH, then:

```bash
R=~/r5gpu; SHA=<sha>; cd $R
gcloud storage cp gs://anthropocene-rl-datasets-508912/model9/r5/code/r5-int-rl-$SHA.tar.gz* . < /dev/null
sha256sum -c r5-int-rl-$SHA.tar.gz.sha256 && mkdir -p code-$SHA && tar -xzf r5-int-rl-$SHA.tar.gz -C code-$SHA && echo $SHA > code-$SHA/COMMIT
mkdir -p $R/scaleout
CODE=$R/code-$SHA/rl setsid nohup bash $R/code-$SHA/rl/ops/r5/scaleout/gpu_queue.sh > $R/scaleout/queue.out 2>&1 < /dev/null &
```

Watch: `tail -f ~/r5gpu/scaleout/queue.log`, `tail -f ~/r5gpu/scaleout/critic/driver.log`,
`grep -E '"kind": "(alarm|rollback|abort)"' ~/r5gpu/scaleout/critic/*/seed-*/stability-critic.jsonl`,
`tail ~/r5gpu/scaleout/memwatch.csv`. A few minutes after launch, check that the fit uses the
GPU (`nvidia-smi`) and the expected threads (`top -H`).

GPU environment (defaults in `gpu_queue.sh`; override by exporting):

| Var | Default | Meaning |
|---|---|---|
| `CODE` | the checkout holding the script | learner checkout (`.../rl`) |
| `PY` | `~/r5gpu/venv/bin/python` | |
| `DATA` / `WORK` | `~/r5gpu/pilot` / `~/r5gpu/scaleout` | staged inputs / outputs |
| `ANTHROPOCENE_DOTNET`, `ANTHROPOCENE_MODEL9_BRIDGE` | `~/r5gpu/bridge/dotnet/dotnet`, `~/r5gpu/bridge/bridge/Anthropocene.Model9Bridge.dll` | `initialize` probes the bridge |
| `BC` | `1` | run the actor BC pilot first |
| `ATTEMPT_ID` | `r5-scaleout-critic-g1-a1` | keep it for resumes; new id for a corrected attempt |
| `SEEDS` / `ARMS` | `0,1,2,3,4` / `privileged-e4:privileged:4:candidate,control-e4:plain:4:control` | |
| `JOBS` / `THREADS` | `1` / `12` | one fit at a time (host RAM, section 5) |
| `LR` | `1e-4` | G1 |
| `FAMILY_WEIGHTS` | `default,under-clothed=2` (cosine decay) | user 27 Oct |
| `INIT_ARGS` | `--qol-encoder shared` | G1 shared encoder (also the campaign default) |
| `CRITIC_ARGS` | `--qol-loss-weight 30 --qol-head-bias-init mean-target --td-step0-targets mc --tail-normalize world-factor --false-terminal-policy mask` | G1 |
| `SEED_GATE` | `1` | seed gate on stability FAIL (population tail excluded, Q1) |
| `POPULATION_TAIL` | `advisory` | Q1: reported, never gating; `gate` restores the 26 Oct rule |
| `GAP_FIT_DATA` | empty (all 8 fit batches) | subset for the gap if memory is short |
| `CONTROL_REPORT` | empty | Q2: empty = the run's own seed-matched control arm is the other-head reference |
| `UPLOAD_EVERY` | `1800` s | background upload of `critic/` |

Fixed by the script: `STAGE=critic DEVICE=cuda EARLY_STOP=1 CRITIC_VALIDATION=1 SEED_RULE=mean-with-floor RESUME_EVERY=100 DETERMINISM_STEPS=50 QOL_REFERENCE_SCALE=<r5-scale-fit8.json>`, 4 epochs, batch 256, grad cap 1, randomized tail required.

### 4.2 CPU VM (anthropocene-trainer-60)

C0, once, after the start (both lines are r5/G0 leftovers; removing the r5 line is approved
"when r5 finishes"; the G0 line belongs to the finished G0 run, see Q6):

```bash
crontab -l > ~/crontab.before-scaleout          # keep a copy
crontab -l | grep -Ev 'g0loop\.sh|r5/loop\.sh' | crontab -
pgrep -af 'loop.sh|drive.sh|g0pipe' || echo "no drivers running"
```

C1, learner checkout with the bridge (same simulator as the teacher data, 8bd480b):

```bash
~/git/checkout.sh <full-sha-of-ops/r5-scaleout> 8bd480b r5s     # -> ~/code/r5s/rl, bridge built
```

C1b, back to Spot (user 28 Oct; from the operator machine, the VM must be stopped):

```bash
DRY_RUN=1 bash ops/r5/scaleout/cpu_spot.sh      # prints: stop, set-scheduling, describe, start
DRY_RUN=0 bash ops/r5/scaleout/cpu_spot.sh
```
It runs `gcloud compute instances set-scheduling anthropocene-trainer-60 --zone us-central1-c
--provisioning-model=SPOT --instance-termination-action=STOP --no-restart-on-failure
--maintenance-policy=TERMINATE --max-run-duration=40h` (Spot needs no automatic restart and
TERMINATE maintenance; if the API still demands it, `EXTRA=--preemptible`). The 8-hour guardrail
is unchanged. `cpu_queue.sh` refuses to start search on a non-Spot VM (`REQUIRE_SPOT=0`
overrides) and, once per learner commit, runs the live kill/resume gate
`tests/test_labor_search_resume_live.py` (~30-60 min) before any search (`RESUME_SELFTEST=0`
skips; not recommended).

C2, launch:

```bash
mkdir -p ~/r5/scaleout
CODE=~/code/r5s/rl setsid nohup bash ~/code/r5s/rl/ops/r5/scaleout/cpu_queue.sh > ~/r5/scaleout/queue.out 2>&1 < /dev/null &
```

C3, relaunch after stops (user-approved @reboot line, 28 Oct). `cpu_queue.sh` installs it
itself once the resume selftest has passed (`INSTALL_REBOOT=1`, default), tagged
`# r5-scaleout-cpu-queue`:

```
@reboot sleep 120; CODE=/home/<user>/code/r5s/rl bash /home/<user>/code/r5s/rl/ops/r5/scaleout/cpu_queue.sh >> /home/<user>/r5/scaleout/queue.out 2>&1 < /dev/null # r5-scaleout-cpu-queue
```
Check with `crontab -l`. The queue is duplicate-safe (`flock` on `~/r5/scaleout/queue.lock`: a
second launch exits 3 without touching the VM). Restarting the VM after a guardrail stop or a
preemption is still manual (no autostart schedule). When search and the panel are done the queue
removes the line itself and writes `~/r5/scaleout/DONE` (a later boot then exits at once without
stopping the VM). Manual removal at any time:
`crontab -l | grep -vF '# r5-scaleout-cpu-queue' | crontab -`.

Dry run of the task list (no work): `CODE=~/code/r5s/rl NORMALIZER=~/r5/actor-normalizer.pt PYTHONPATH=~/code/r5s/rl/src ~/m9-bench/venv/bin/python ~/code/r5s/rl/ops/r5/scaleout/cpu_sched.py --dry-run | head`.
Watch: `~/r5/scaleout/status.json` (counts, mean hours per task, projected remaining wall hours),
`sched.log`, `memwatch.csv`, `logs/search-*.log`.

CPU environment (defaults in `cpu_queue.sh` / `cpu_sched.py`):

| Var | Default | Meaning |
|---|---|---|
| `CODE`, `PY` | `~/code/r5s/rl`, `~/m9-bench/venv/bin/python` | clean checkout required (search refuses dirty identities) |
| `WORK`, `NORMALIZER` | `~/r5/scaleout`, `~/r5/actor-normalizer.pt` | |
| `MAX_SLOTS` | `16` | concurrent processes (search chunk = 1, panel world = `PANEL_JOBS`) |
| `MIN_FREE_GB` | `20` | no launch below this MemAvailable |
| `FIT_WORLDS` / `RESERVED_WORLDS` | `90100-90153,90160-90165` / `90154-90159` | |
| `FIT_SPLITS` | `0:25:8 25:50:8` | first:turns:roots per chunk → 16 roots/world over 2 strata (with root-level resume, small chunks no longer bound the loss) |
| `RESERVED_SPLITS` | `0:13:8 13:25:8 25:38:8 38:50:8` | 32 roots/world |
| `CANDIDATES` | `14` | M; 2 plans x 2 rows are reserved from M, so 10 single-step candidates as in G0 |
| `HORIZONS` / `KEEP` / `REEVAL_STREAMS` | `1,3` / `all,3` / `2` | G0 settings; re-evaluation scope final-survivors (payload default) |
| `CONTROL_FRACTION` / `PLAN_CANDIDATES` / `PLAN_STEPS` / `PLAN_FRACTION` | `0.25` / `2` / `3` / `0.25` | plan horizon max(3, 3+1) = 4 |
| `PERSONALITY` / `PERSONALITY_KEY` | `dirichlet:1.0` / `4242` | the teacher collection's lineage stream (same personalities as the V data) |
| `PLAYERS`, `WIDTH`, `HEIGHT`, `SEPARATION` | `16`, `128`, `64`, `9` | the r5 reset (G0 search used 8 players) |
| `ACTOR_GCS` / `ACTOR` | `$SCALEOUT_GCS/actor-pilot` / `~/r5/scaleout/actor/actor-bc-pilot.pt` | fetched when `DONE` appears; sha256-checked |
| `ACTOR_WAIT_H` | `6` | give up on actor-gated tasks after this long idle |
| `ACTOR_ROOTS` | `0` | 1: also `--source actor` chunks (`ACTOR_FIT_SPLITS 0:25:2 25:50:2`, `ACTOR_RESERVED_SPLITS 0:25:4 25:50:4`, `ACTOR_SAMPLES 4`); see Q4 |
| `PANEL`, `PANEL_WORLDS`, `PANEL_TURNS`, `PANEL_MODES`, `PANEL_JOBS` | `1`, `2000-2023`, `50`, `h12,deterministic,stochastic`, `4` | never 3000-3059 (refused) |
| `SEARCH_EXTRA` | empty | appended to every labor_search command |

### 4.3 Actor BC pilot on the CPU instead (only with `BC=0` on the GPU)

```bash
C=~/code/r5s/rl; CODE=$C PY=~/m9-bench/venv/bin/python DATA=~/r5 NORMALIZER=~/r5/actor-normalizer.pt \
PRIV_NORMALIZER=~/r5/privileged-normalizer.pt OUTD=~/r5/scaleout/actor-pilot DEVICE=cpu THREADS=16 \
ANTHROPOCENE_DOTNET=/usr/bin/dotnet ANTHROPOCENE_MODEL9_BRIDGE=$C/integration/bin/model9/Anthropocene.Model9Bridge.dll \
setsid nohup bash $C/ops/r5/scaleout/actor_bc_pilot.sh > ~/r5/scaleout/bc.out 2>&1 < /dev/null &
```
(it uploads to `scaleout/actor-pilot/`, where the scheduler picks it up; lower `MAX_SLOTS` by ~4
while it runs).

### 4.4 Later: the A fit and gate (d), once payloads exist (GPU VM, same `OUT`)

```bash
S=~/r5gpu/search; gcloud storage rsync -r gs://anthropocene-rl-datasets-508912/model9/r5/scaleout/search $S < /dev/null
FIT_PAYLOADS="$(ls -d $S/fit-h12/*/w*/world-* | tr '\n' ' ')" RESERVED_PAYLOADS="$(ls -d $S/reserved-h12/*/w*/world-* | tr '\n' ' ')" \
STAGE=all ... (the same variables as gpu_queue.sh step 4) bash $CODE/tools/r5_critic_acceptance.sh
```
This reuses every `critic.pt`, fits A per (arm, seed), writes `evidence.json` and
`gate-receipt.json`. (Not scripted yet: it depends on Q3 and the search outcome.)

## 5. RAM and disk sizing

Data in memory = on-disk shard size (uncompressed `.npy`; `labor_activity_features` is 88% of
it). Before bd800d0 a load peaked at ~2× (per-shard arrays plus the concatenation).

| Process | Steady host RSS (estimate) | Peak | Notes |
|---|---|---|---|
| Critic fit, 60 worlds + per-epoch validation on 20 | 26.3 + 9.5 + ~4 (torch, CUDA ctx, model, targets) ≈ **40 GB** | ~42 GB | before bd800d0: **~55 GB** at the fit-data load; with ~61 GB usable on the L4 host only one fit fits either way |
| evaluate (cal 9.5 + full fit data for the gap 26.3) | ~38 GB | ~40 GB | `GAP_FIT_DATA="$DATA/fit8-b01 $DATA/fit8-b02"` drops it to ~21 GB |
| QoL scale (8 fit batches) | ~27 GB | ~28 GB | |
| BC pilot (b01..b04, 14 GB) | ~16 GB | ~17 GB | |
| GPU memory per fit | 1.5-4 GB | | of 24 GB |
| Search worker (16 players, 2 bridges + forks) | unmeasured; collection workers were 2-6.8 GB | | `MIN_FREE_GB` admission; watch `memwatch.csv` in the first 2 h and raise `MAX_SLOTS` to 18-20 if peak RSS stays ≤ 5 GB |
| Panel world (4 processes) | ~4 × 4-5 GB | | |

These are estimates from shard sizes and code reading, not measurements. `memwatch.csv`
(both VMs) records MemAvailable, the summed and largest python RSS and the bridges' RSS every
1-2 min; the queue logs `peak_python_rss_gb` at exit. **Record the seed-0 critic fit's peak
as the measured 60-world RSS.** Two fits at once (JOBS=2) need ~80 GB: not on this host
(g2-standard-32 has 128 GB but exceeds the 80-CPU quota while the CPU VM runs).

Disk: GPU needs ~36 GB of inputs (some already in `~/r5gpu/pilot`), ~3 GB of outputs
(checkpoints ~60-100 MB each, resume states excluded from upload); the script requires 40 GB
free. CPU: search outputs ~1-2 GB (G0's 480 roots were 0.2 GB without payloads), panel
reports < 1 GB; 40 GB free required.

## 6. Wall time and cost (estimates)

GPU (L4, from the G1 pilot: 1.04 s/step plus epoch work; 4 epochs = 3,944 steps at 60 worlds):

| Step | Estimate |
|---|---|
| Staging + QoL scale | 15-30 min |
| BC pilot: 136,472 rows, 534 steps/epoch × 25 epochs = 13,350 steps at ~0.35-0.6 s/step (unmeasured on L4) | 1.3-2.2 h + 10 min fidelity |
| One critic fit: load ~5 min + 3,944 steps (~68 min) + 16 TD refreshes and 4 validations over 91k cal rows (~20 min) | ~1.6 h |
| evaluate per fit | ~10-20 min |
| determinism probe (2 × 50 steps + loads) | ~15 min |
| **10 fits (seed gate passes)** | **~18-20 h** |
| **Total GPU queue** | **~20-23 h**, ≈ $25-30 on-demand at ~$1.15/h (g2-standard-16, estimate) + ~$3-5 GCS egress (asia-northeast1 → us-central1) |

If the seed gate trips (a non-tail stability FAIL of the candidate's seed 0), the queue ends after ~4 h of critic work (~$7).

CPU search (from G0: 42 resolves/root, 20.8 s/resolve at 8 players, 20 jobs):

| Item | Estimate |
|---|---|
| Resolves/root: search 26 (M=10 singles, as G0) + final-survivor re-evaluation 30 + controls on 25% (~11) + plans on 25% (2 plans × 3 streams × 13 resolves × 0.25 ≈ 20) | **~87** (G0: 42) |
| s/resolve at 16 players, 128×64 | 30-55, central ~35 (collection: 1.1-2.4 min per 2-resolve decision) |
| Per root | ~50 min (35-80) |
| Roots | 960 fitting + 192 reserved = 1,152 |
| Main path per chunk (drive to the chunk's last decision, ~2 resolves/decision) | ~0.6 h avg × 144 chunks |
| **Total** | **~1,050 process-hours (range ~650-1,700)** |
| Wall at 16 slots | **~70 h (45-110)**; PHASE1 (reserved + 8 roots/world) ≈ 55% of it |
| Panel (24 worlds × 4 runs × 50 decisions) | ~60 process-hours, ~4-5 h with priority |
| Cost on Spot (~$0.8-1.1/h, estimate; verify) | **~$60-90** central for ~70 h, plus preemption losses (7); on-demand would be ~$180 |

Cheaper knobs (each is a deviation from the approved scale, so a user decision):
`PLAN_FRACTION=0` (−~22%), `FIT_SPLITS="0:25:4 25:50:4"` (8 roots/world, −~40% of fitting),
`CANDIDATES=10` (6 singles, −~5%).

The scheduler's `status.json` replaces these guesses within ~6 h: it reports mean hours per
task per group and the projected remaining wall time.

## 7. Failure and resume

Rerunning the same launch command is always the resume step: finished outputs are skipped.

| Event | What survives | Operator action |
|---|---|---|
| GPU max-run (20 h) stop, preemption, guardrail | finished fits; a running critic fit resumes bit-for-bit from `critic.resume.pt` (every 100 steps); evaluate reruns; BC resumes from `actor-bc-pilot.pt.partial` (every 500 steps; Adam, shuffle and family-weight decay restart: disclosed pilot approximation) | start the VM, rerun 4.1's last line (same `ATTEMPT_ID`) |
| CPU 8-hour guardrail, Spot preemption | finished chunks and, inside a running chunk, every completed decision boundary: an interrupted chunk resumes from its newest checkpoint (`search_resumed_from_checkpoint` in its log), losing at most the in-flight decision and its root (~1 h per slot, ~0.5 h on average; ~8 process-hours per stop at 16 slots, ~6% of an 8-h window). Panel worlds rerun from reset (≤ ~1.5 h each) | start the VM, rerun C2. A refused checkpoint (`resume_refused` in the log) redoes that world from its reset and is recorded in its meta |
| A stability abort / seed-gate trip | `seed-gate.json`, `evidence-aborted.json`, the receipt (FAIL) | escalate per the manifest (more than one seed of an arm, any population-tail trigger) |
| A task fails twice (CPU) | `~/r5/scaleout/FAILED/<task>`; the rest continue | read `logs/<task>.log`; rerun C2 after the fix (failed markers are not retried automatically: delete the marker file to retry) |
| Scheduler killed but children alive | `cpu_queue.sh` refuses to start | `pkill -f "labor_search .*--out $HOME/r5/scaleout/"` (and the panel pattern), then rerun C2 |
| OOM-killed worker | treated as a failed attempt and retried once with `--resume` | lower `MAX_SLOTS` or raise `MIN_FREE_GB` |
| Wrong config discovered mid-run | search refuses to resume a chunk whose source record (code, bridge, options) differs | new `WORK` (or move the affected `search/<stage>` dir aside) |

### 7.1 Successor-row tasks (4 Nov 2026, approved option A)

The stage-1 A fit needs search export-v2 shards (`labor_search --successor-rows`). `cpu_sched.py`
appends `--successor-rows` to every task named in `WORK/SUCC_TASKS` (one key per line, e.g.
`search:fit-h12/t25-50/w90116`) and runs it with `PYTHONPATH` from `WORK/SUCC_PYTHONPATH`, a
`src` directory whose `labor_search.py` must accept the flag. Without `SUCC_TASKS` nothing changes.
The live run listed the 38 fit tasks that had not started (`succ_tasks_20261104.txt`) and used
code 801b02b. The earlier fit tasks remain v1, and stage-1 refuses them.

Run the patched scheduler from a copy outside the learner checkout (on trainer-60:
`~/r5/scaleout-ops-succ/`). `cpu_queue.sh` refuses a dirty checkout, and its EXIT trap then
shuts the VM down. The `@reboot` line that `cpu_queue.sh` installs points at whichever copy
launched it; do not remove it while the queue runs, because only a fresh queue start reinstalls it.

## 8. Expected outputs (`gs://anthropocene-rl-datasets-508912/model9/r5/scaleout/`)

| Path | Content |
|---|---|
| `r5-scale-fit8.json` | QoL reference-scale artifact on all 60 fitting worlds |
| `critic/critic-receipt.json` | receipt with `configuration.stage = critic`, `critic_stage` (V-only status per arm, per seed, deferred A checks, not-passing checks, determinism, arm comparison); `machine_verdict` stays HOLD until stage all |
| `critic/<arm>/seed-<s>/` | `checkpoint-000.pt`, `critic.pt` (+ `.settings.json`), `stability-critic.jsonl`, `evidence-critic.json`, `fit.log`, `timings.json` |
| `critic/seed-gate.json`, `critic/determinism/determinism-critic.json`, `critic/driver.log` | |
| `actor-pilot/` | `actor-bc-pilot.pt` (+ `.sha256`), `stability-bc.jsonl`, `bc.log`, `fidelity-{cal8-b01,cal8-b02,cal8-b03,fit8-b05}.json`, `DONE` |
| `search/{reserved-h12,fit-h12}/t<first>-<turns>/w<world>/` | labor_search datasets: `world-<seed>/` shards with `advantage-candidates.pt` (labor-advantage-candidates-v1), `search_report.json`, `dataset.json` |
| `panel/world-<seed>.json`, `panel-report.json`, `panel-report.txt` | per-world autonomous reports; merged report with the manifest gate (vs H12) and brief S1 (vs H) |
| `RESERVED_H12_DONE`, `PHASE1_DONE`, `SEARCH_H12_DONE`, `PANEL_DONE` | milestones |
| `gpu-logs/`, `cpu-logs/` | queue logs, `memwatch.csv`, scheduler `status.json`, `durations.jsonl`, task logs |

### Actor pilot metrics and gates

Indicative only (4 of 8 fit batches, 1 seed; manifest §5: an initializer, not a gate pass).

- **Manifest §8 (primary):** vs H12, welfare and person-years ≥ 0.95 aggregate and ≥ 0.90 per
  world, deterministic and stochastic separately, no starvation extinction beyond H12's
  (`panel-report.json: manifest_gate_vs_h12`).
- **Brief S1 (reported):** vs H, ≥ 0.90 aggregate, ≥ 0.85 per world, both modes, no additional
  starvation extinction (`brief_s1_gate_vs_h`, the evaluator's built-in gate).
- **Interface quality:** H12/H welfare and person-years (G0: 0.965 / 1.082).
- **Same-state fidelity (manifest §6):** specialty and activity L1, per-specialty shares,
  movement/fission/trade-gate accuracy and focus activities (Craftsperson housing and housing
  material: teacher ~32% each, arm3-v2 ~3%) on the calibration worlds and the unseen fitting
  batch fit8-b05 (`fidelity-*.json`). No r5 development-world teacher bank exists, so the
  §6 "development roots" support ledger is not available from this pilot.
- The 24 × 50 panel is a scale-out test; comparisons with G0's 8 × 8 panel are indicative only.

## 9. Open questions (decisions before launch)

- **Q1, decided (user 28 Oct): population tail is advisory for r5.** `run --population-tail
  advisory` (default; `POPULATION_TAIL` in the shell driver, `advisory` in `gpu_queue.sh`,
  PREREGISTERED `population_tail_policy_r5`): the check is reported per fit as
  `advisory:population-tail` (FLAG and `escalate` on a FAIL, still a user-escalation item) and
  never fails stability, the receipt or the seed gate. Every other stability check still gates
  and still trips the seed gate. (The G1 pilot FAILED it: upper_excess 0.032 > 0.01.)
- **Q2, decided (user 28 Oct): the other-head reference is the run's own control arm.** With no
  `CONTROL_REPORT` (critic3 carries the G0 normalizer, so it has no r5 version), each candidate
  fit's other heads are compared with the seed-matched plain control arm's evaluation (same
  calibration worlds, evaluator and units); control-arm fits are their own reference. The receipt
  records `regression_reference` per fit. A supplied `CONTROL_REPORT` still wins.
- **Q3: actor checkpoint for actor roots.** The G0 final actor
  (`datasets/labor-g0-20261024/final/final-actor.pt`) cannot be used: its normalizer (digest
  d49f9d35…) differs from the r5 actor normalizer (fcc769c3…, sha 3c8c5ca3…), the action-12
  context fix changed core fields, and `labor_search` refuses a mismatched actor. The only r5-
  compatible actor will be this BC pilot (or a later r5 actor). So the default search is H12
  roots without actor candidates (`--actor-samples` unused); G0's wins were mostly actor samples,
  so actor-visited roots and actor candidates should follow from the pilot actor (`ACTOR_ROOTS=1`)
  or the first r5 actor. Approve the pilot actor for that, or wait?
- **Q4: "16 roots per fitting world"** is read as 16 H12 roots; actor roots (if Q3) come on top
  (default 4/world, 8/reserved world). Scenario roots: the coverage audit found no gaps, so no
  scenario roots exist; nothing is scheduled for them.
- **Q5, decided (user 28 Oct): root-level resume, Spot, guardrail kept.** Implemented in 71782ee;
  the VM goes back to Spot via `cpu_spot.sh` (C1b). Each stop still needs a manual restart
  (no autostart schedule is attached; adding one is persistence and needs the user's OK).
- **Q6: the G0 `@reboot` line** (`g0loop.sh`) is still in the CPU crontab per the last snapshot;
  it shuts the VM down when `~/r5/ACTIVE` is absent. Removing it is in C0; confirm.
- **Q7: budget.** Search at the approved scale is the dominant cost (~$180-200 on-demand central
  estimate, ~70 h). Cheaper knobs are in section 6.
- **Q8: design choices made here, for review:** disjoint decision strata per chunk
  (fitting `0-25, 25-50`, 8 roots each; reserved `0-13, 13-25, 25-38, 38-50`, 8 each, so its 6
  worlds run as 24 parallel chunks) instead of one uniform draw (needed
  for unique root ids across chunks, and it stratifies roots over the window); `CANDIDATES=14`
  (keeps G0's 10 single-step candidates next to the 2 plans); personality key 4242 (the teacher
  collection's); BC 25 epochs (the initial half of a 50-epoch program, the S1 checkpoint) on
  fit8-b01..b04; "early stop on" is read as the live stability early stop (`EARLY_STOP=1`), and
  best-validation checkpoint selection (`--select-best-validation`) is not added: the G1 pilot's
  exact command line is not in its uploaded logs, so confirm (add it via `CRITIC_ARGS` if wanted).
  `~/git/checkout.sh <rl> <sim> <name>` is assumed to work as it did for r5e/r5f.
