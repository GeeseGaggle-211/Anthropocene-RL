# CPU-VM operator handoff: r5 teacher collection (27 Oct 2026, 03:30Z)

You are taking over routine operation of `anthropocene-trainer-60`. The workload is running. Don't change it except as described here, and report to the co-architect anything this document doesn't cover.

## 1. VM facts

- **Name and zone:** `anthropocene-trainer-60`, zone `us-central1-c`, project `anthropocene-508912`. c3d-highcpu-60, 115 GB RAM. The boot disk was resized online to 200 GB.
- **Scheduling:** on-demand. Read back at 03:30Z: `provisioningModel: STANDARD`, `preemptible: false`, `maxRunDuration: 144000s` (40 h), `instanceTerminationAction: STOP`, `automaticRestart: false`.
  - The last start was 02:45:17Z on 27 Oct, so max-run would stop the VM at about **28 Oct 18:45Z**.
  - In practice the 8-hour guardrail (section 4) stops it first, and each start restarts the 40 h clock.
  - The switch from Spot used `set-scheduling --no-preemptible --provisioning-model=STANDARD`, in place. The VM was not recreated.
- **Resource policy:** `g0-autostart5` is attached (start every 2 h at :05; ends 28 Oct 12:00Z). Leave it. The co-architect detaches it when collection is done.
- **SSH command (PowerShell or Git Bash):**
  `gcloud compute ssh anthropocene-trainer-60 --project=anthropocene-508912 --zone=us-central1-c --tunnel-through-iap --strict-host-key-checking=yes --quiet "--command=..."`
- **Pinned host key:** `SHA256:qv55HFWP7NUJyGC4CvB6H8vvFO8YeNANHgAeYVi0zfE`. It is unchanged; the instance was never recreated. If a host-key prompt ever appears, STOP and report.
- **Remote home:** `/home/suyashbhatt11_gmail_com`. pscp needs absolute remote paths.

**Ssh hygiene, which is mandatory:**
- Never put a `pkill`/`pgrep` pattern on the ssh command line; it matches its own shell. Put such commands in a script file.
- Ship multi-line scripts base64-encoded: `echo <b64> | base64 -d > ~/r5/x.sh; bash ~/r5/x.sh`.
- PowerShell mangles quotes in `--command=`, so write scripts locally and send them as base64.
- Launch background jobs with `(setsid nohup bash x.sh > x.out 2>&1 < /dev/null &)`.
- Never print credentials.

**Crontab (user), with two `@reboot` lines:**
```
@reboot sleep 90; bash /home/suyashbhatt11_gmail_com/g0loop.sh > /dev/null 2>&1
@reboot sleep 120; bash /home/suyashbhatt11_gmail_com/r5/loop.sh > /dev/null 2>&1
```
- The G0 loop is harmless: `~/g0/ALLDONE` exists, so `g0loop.sh` exits immediately.
- `~/g0pipe.sh` has guarded shutdown lines (13 and 105) that skip while `~/r5/ACTIVE` exists. **Never delete `~/r5/ACTIVE`.**
- As a precaution, the pending G0 `@reboot` shell has been killed at each manual restart (section 4).

## 2. ~/r5 layout and semantics

**Driver: `~/r5/drive.sh`**, md5 `3d915dc30d7444dca544be83b447fbef`. Local copy: `cpu-operator/drive-r5e4.sh`.
- It is idempotent and runs the collection from `C=~/code/r5e/rl` (learner b6a1290), with `PYTHONPATH=$C/src`, `OMP_NUM_THREADS=1`, the r5e bridge and `~/m9-bench/venv/bin/python`.
- It takes its own lock, `exec 9>~/r5/drive-main.lock; flock -n 9 || exit 0`. Batch subshells are started with `9>&-`, so killing the driver's main process frees the lock while its batches keep running.
- `in_use <batch>` skips any batch whose `--out` directory is already being written by a running collector. This prevents duplicates after a driver restart.
- **Batch command** (identical for every batch except `--worlds` and `--out`; heldout50 adds `--held-out-test`):
  `python -m anthropocene_rl.autonomous_v7.labor_teacher_data --normalizer-checkpoint ~/r5/actor-normalizer.pt --turns 50 --players 16 --width 128 --height 64 --separation 9 --personality dirichlet:1.0 --personality-key 4242 --return-tail randomized --jobs 8 --worlds <set> --out ~/r5/<batch> [--resume]`
- **Resume:** if `~/r5/<batch>` exists, the driver adds `--resume`, which continues each unfinished world from its newest per-decision checkpoint. The checkpoints live in `<out>/_work/world-<seed>/ckpt-*`, and the run identity is in `<out>/collection.json`. The driver never moves or deletes an output directory.
- **Launch rule:** the next batch starts only when all three hold: batch parents < MAXB, active workers ≤ MAXW, and MemAvailable ≥ MINMEM GB.
  - The values come from `~/r5/launch.conf` (currently `MAXB=3 MAXW=8 MINMEM=60`), which is re-read before every check. A rule change needs no restart.
  - "Active" means a worker used CPU during a 10 s sample, so idle pool workers don't count.
  - MAXW=8 is the memory ceiling: 16 active workers at about 6.8 GB tail peak is the most the VM holds.
- **End of run:** after all 12 datasets exist, the post-collection stage runs (section 3), then the driver writes `COLLECT_DONE` and runs `sudo -n shutdown -h now "r5 collection complete"`.

**Loop: `~/r5/loop.sh`**, md5 `b8dd2095451a2f41af82f4db49d64052`. Local copy: `cpu-operator/loop11.sh`.
- Every 5 minutes until `COLLECT_DONE`, it starts `~/r5/memwatch.sh` if that is not running, then runs `bash ~/r5/drive.sh`. That driver run exits at once if another driver holds the lock.

**Lock files:**
- In use: `drive-main.lock`.
- Stale, ignore: `drive.lock`, `drive2.lock` and `drive3.lock`. Older batch subshells inherited and held these. The b03/b04 subshells, started by an earlier driver, still hold `drive3.lock`; harmless.

**Markers in ~/r5:**
- `ACTIVE`: present. It keeps G0 from shutting down or detaching the schedule. Never remove it.
- `STOP`: absent. If present, every driver launch exits at once. It is written on a hard failure (for example "collection incomplete"); only the operator clears it, after fixing the cause.
- `<batch>.FAILED`: none present. The driver skips that batch. It is written when a batch's collector exits non-zero or without a dataset.json. Recovery is in section 5.
- `POST_APPROVED`: present. It lets the approved post-collection stage run automatically.
- `COLLECT_DONE`: absent. It is written at the very end; the loop and driver then stop, and the VM shuts down.

**Other files in ~/r5:**
- `driver.log`: the driver's event log; read this first.
- `<batch>.log`: collector stdout/stderr. During a world's window there is one JSON progress line per decision. **During the randomized tail it prints nothing** until the world finishes; that is normal.
- `memwatch.csv`: per-worker RSS samples every 2 minutes (tree = python plus its 2 bridges).
- `alert.sh`, `alert.state`: the alert checker the poller calls (section 6).
- `scen.sh`, `scen2.py`, `food3.py`: the scenario and food-trade checks (section 6).
- `actor-normalizer.pt`: the actor normalizer, sha256 prefix 3c8c5ca3…, fitted on 90100-90107, 25 turns. It is also staged at `gs://anthropocene-rl-datasets-508912/model9/r5/gpu-collect/`.
- `fit8-b02.old-c9fc899/`: the 7 old-code shards of the first b02 attempt. **Keep it as evidence; don't delete it.**
- `fit50-*.partial-*`, `fit8-b0?.partial-*`: earlier interrupted attempts, kept. `fit50-b02.partial-1793031500/world-90110` is the GPU parity reference shard.

**Checkouts (all with simulator 8bd480b):**

| Checkout | rl commit | Bridge dll sha256 | Used for |
|---|---|---|---|
| `~/code/r5e` | b6a129054c6e… | 32c967a96b8752c68a8fc00702a8b0acdf4eacffabb4911e9772eb258771629b | All collection and post-collection. Has resume, needs resumeContinuation ≥ 3 |
| `~/code/r5c` | c9fc8995c6fa… | b5439c49aff3aedbb083b9565c3de3769a04f9998289eeb02cfb5e0ded784463 | Normalizer fit and fit8-b01; old code, no resume |
| `~/code/r5d` | 51c139a07abb… | cfd532a8eeb285626af62c19044e82e242c9f75eb08457fd694c2e7827f05873 | `labor_scenarios` (scen.sh) and later search |

**Batch queue and status at 03:30Z.** The queue order is fit8-b01 … b08, cal8-b01 … b03, then heldout50.

| Batch | Worlds | Status |
|---|---|---|
| fit8-b01 | 90100-90107 | **Published** (dataset.json; c9fc899 collector) |
| fit8-b02 | 90108-90115 | **Full r5e rerun pending.** Queued next; waits for the launch rule (no output dir yet, so it starts without `--resume`) |
| fit8-b03 | 90116-90123 | Running (r5e), resumed 02:47Z. Decisions 42-50 at 03:30Z |
| fit8-b04 | 90124-90131 | Running (r5e), resumed 02:49Z. Decisions 35-50 at 03:30Z |
| fit8-b05 | 90132-90139 | Queued |
| fit8-b06 | 90140-90147 | Queued |
| fit8-b07 | 90148-90153, 90160-90161 | Queued (comma list) |
| fit8-b08 | 90162-90165 | Queued (4 worlds) |
| cal8-b01 | 90180-90187 | Queued |
| cal8-b02 | 90188-90195 | Queued |
| cal8-b03 | 90196-90199 | Queued (4 worlds) |
| heldout50 | 90154-90159 | Queued, with `--held-out-test` |

- **Timing:** a world takes about 2.1-6.2 h (50-decision window plus a randomized tail of about 47-96 decisions), and a batch takes as long as its slowest world.
- The memory rule allows about 2 batches at once, so expect about 6-8 h per pair. **The ETA for the rest is about 30-35 h of collection plus about 1 h of post-collection**, and longer if guardrail stops land mid-tail.

## 3. Post-collection (approved; runs automatically from the driver, on r5e)

```
cd ~/code/r5e/rl; PYTHONPATH=src OMP_NUM_THREADS=8 python -m anthropocene_rl.autonomous_v7.labor_training fit-privileged-normalizer \
  --dataset ~/r5/fit8-b01 … --dataset ~/r5/fit8-b08 \
  --calibration-dataset ~/r5/cal8-b01 --calibration-dataset ~/r5/cal8-b02 --calibration-dataset ~/r5/cal8-b03 \
  --report ~/r5/privileged-normalizer-report.json --out ~/r5/privileged-normalizer.pt
python -m anthropocene_rl.autonomous_v7.labor_training privileged-dead-mask-audit \
  --dataset ~/r5/fit8-b01 … --dataset ~/r5/fit8-b08 --out ~/r5/privileged-dead-mask-audit.json
```
- heldout50 is excluded from both by design.
- After these, the driver writes `COLLECT_DONE` and shuts the VM down. Then report milestone (c) to the co-architect.
- Known downstream blocker, assigned to an implementer: `pretrain-critic` refuses more than one teacher `--dataset` (labor_training.py:1142-1150).

## 4. The 8-hour guardrail (user's cost control; do not modify)

- A systemd timer from the instance startup-script, `anthropocene-auto-shutdown.timer` (`OnBootSec=8h`), runs `shutdown -h now "Anthropocene 8-hour cost guardrail"`. GCE logs this as `compute.instances.guestTerminate`.
- **The next fire is 27 Oct 10:45:25Z**, then 8 h after each boot.
- **Do not modify the timer or the startup-script metadata.** The user decides whether to replace it.
- **Restart procedure, to do at once; don't wait for the autostart slot:**
  1. Confirm the stop reason: `gcloud compute operations list --project=anthropocene-508912 --filter="targetLink~anthropocene-trainer-60" --sort-by=~insertTime --limit=3` should show `guestTerminate`. After boot, `journalctl -b -1 | grep 'powering down'` gives the message.
  2. `gcloud compute instances start anthropocene-trainer-60 --zone us-central1-c --project anthropocene-508912`.
  3. SSH with strict host-key checking. If any prompt appears, STOP and report.
  4. **Within 90 s of boot**, run by script: `for p in $(pgrep -f 'g0loop[.]sh'); do kill $p; done`. This kills the pending `sleep 90; bash g0loop.sh` shell, as a precaution.
  5. About 120 s after boot, `loop.sh` starts the driver. Confirm in `driver.log` that each interrupted batch logs `batch <name> start … --resume`.
  6. Check `grep -E 'resumed_from_checkpoint|collection_resume_fallback' ~/r5/<batch>.log | tail` per batch. Report each world's `resumed_from_checkpoint` and `in_tail`, and any `collection_resume_fallback`.
  7. Reapply OOM priority to every r5e batch's workers and bridges (see `~/r5/res.sh`, which does exactly this and reads it back):
     ```
     P=<batch parent pid: the labor_teacher_data process with --out ~/r5/<batch> whose parent is not labor_teacher_data>
     for w in $(pgrep -P $P); do for p in $w $(pgrep -P $w); do echo 300 | sudo -n tee /proc/$p/oom_score_adj; done; done
     ```
- Reapply +300 to any newly launched batch too; new processes start at 0.

## 5. Memory and OOM policy

- Per worker tree: about 2-3 GB early, about 4 GB at decision 25, and **about 6-6.8 GB at the tail peak**. MAXW=8 keeps the projected peak under 115 GB.
- The 60 GB launch check only applies at launch; tails keep growing afterwards.
- The low-memory alert fires below about 15-18 GB available.
- r5e batch processes carry `oom_score_adj +300`, so the kernel kills a resumable process first.
- **If an OOM kill happens** (`journalctl -k | grep -i 'out of memory\|oom-kill'`):
  - Report it at once.
  - If a batch then fails (`FAIL batch <name>` in driver.log and `<name>.FAILED`), recover with `--resume` once memory allows: delete `~/r5/<name>.FAILED`. The next driver pass reruns that batch's exact command with `--resume`, because the output directory exists.
  - The running driver only checks markers when it reaches a batch. If it has already passed that batch, restart the driver: kill only the driver's main process (the `bash ~/r5/drive.sh` holding `drive-main.lock`, with the loop as its parent). Batches keep running, and the loop relaunches the driver within 5 minutes.
  - Don't kill anything else proactively.
- Never stop the VM while a batch from an older non-resumable checkout is running; none is running now.

## 6. Monitoring

- **Cadence:**
  - every 20 minutes while worlds are in their tails, and while a guardrail stop is expected;
  - otherwise every 60 minutes;
  - each check covers available memory, the largest worker RSS, progress and tracebacks.
- **Alerts:** report at once on any of these:
  - VM not RUNNING;
  - available memory below 15 GB;
  - a kernel OOM kill;
  - a traceback in a batch log;
  - `FAIL batch`, `FAIL fit…`, `incomplete` or `STOP` in driver.log.
- **Report on each batch completion**, with an updated ETA.
- **Checker:** `~/r5/alert.sh` prints `ALERT …` lines for low memory (< 18 GB), OOM lines in the last 25 minutes, tracebacks, driver failures, STOP, newly published batches and COLLECT_DONE. It then prints `TAILS=<n> avail=…GB` and `LOAD=… W=<collector process count>`, plus the last batch events.
  - `alert.state` records events already reported. `fit8-b01` and `driverfail` checks are de-duplicated this way.
- **Poller:** `cpu-operator/poll3.sh` loops on alert.sh, writes a heartbeat line every check, and exits on any ALERT or on the VM not RUNNING.
  - Its first line `cd`s into my session scratchpad. Change that path before reusing it.
  - Run exactly one poller.
- **scen.sh** (`~/r5/scen.sh <dataset dir>`, runs from `~/code/r5d`): `labor_scenarios weights` plus the food-trade counts. Its output for fit8-b01 is in `cpu-operator/scen-fit8-b01.out`.

## 7. Local files (all in `C:\Users\suyas\Documents\ChatGPT\worktrees\model9-data\g0\cpu-operator\`)

| File | Content |
|---|---|
| tally.log | Preemption, stop and batch tally since 26 Oct 08:05Z |
| poll.log | Every poller check result |
| vm-scheduling.yaml | Scheduling and resource policies at 03:30Z |
| poll3.sh | Single heartbeat poller |
| alert.sh | Alert checker; the VM copy is ~/r5/alert.sh |
| memwatch.sh | Memory watcher; the VM copy is ~/r5/memwatch.sh |
| drive-r5e4.sh | Copy of the installed drive.sh |
| loop11.sh | Copy of the installed loop.sh |
| scen.sh, scen2.py, food3.py | Scenario weights and food-trade checks |
| scen-fit8-b01.out | fit8-b01 scenario table and food-trade counts |
| smoke.sh, smoke2.sh | The r5e resume smoke test and its bit-identity check |
| ev.out, ev2.out, ev5.out | Evidence for the 02:40Z guardrail shutdown |
| res.out | b03/b04 resume points after the 02:45Z restart |
| agg.py | G0 summary aggregator (wrote ../g0-final-summary.json) |

## 8. Open items

1. **fit8-b02 full rerun:** queued, not started. It needs the launch rule to allow it (b03/b04 workers draining).
2. **Tail resume is untested.** The 02:45Z restart resumed only in-window checkpoints, at decisions 22-36. The 10:45Z guardrail stop will likely be the first tail resume; report each world's `resumed_from_checkpoint`/`in_tail` and any `collection_resume_fallback`.
3. **Detaching g0-autostart5** after collection: the co-architect does this. Don't detach it yourself.
4. **Guardrail replacement** is pending the user's decision.
5. **Food trade finding** (for the user): in fit8-b01, the food-trade family has 0 rows.
   - Food imports appear in 31 of 33,051 rows.
   - The stored "imported" value saturates at the normalizer guard (about 1.72), because that column's std is at the 0.1 floor.
6. **pretrain-critic multi-dataset blocker:** with an implementer.

## 9. Local tasks stopped at handoff

| Task ID | What it was |
|---|---|
| `b1zuscg2r` | The single heartbeat poller (poll3.sh) |
| `bf1xphxyv` | Poller restart attempt; it had already exited with a sed error |
| `bcrgiu7ug` | Earlier poll3 instance |
| `b6tyvcxvi` | Earlier poll3 instance |
| `bucvc9pjx` | Stale poll2 instance |
| `bfsci8ewm` | Stale poll2 instance |
| `b4fnws5im` | Earlier poll2 |

- Other background sleeps and waiters (`b8189fdyr`, `bu4gavvsy`, `bu9g3i5as`, `bnkvbg016`, `b0sbtvzf7`, `bfmg0w04r`) had already completed.
- No local background tasks are running now.
- On the VM, nothing was stopped: the r5 loop (`bash ~/r5/loop.sh`), the driver (pid 6115), the b03/b04 batches and `~/r5/memwatch.sh` are all running.

## 10. r5f checkout and drive.sh pinning (27 Oct, operator addendum)

- `~/code/r5f`: rl e6f8b67566fe4a133f824d8be3a42ccaf955ed5c, sim 8bd480b, built 12:38Z.
- Bridge dll sha256, r5f: `8d318069b626e84387c71aebad931c55368296a1ec2736f0c4d2c1885aeffaae`
- Bridge dll sha256, r5e: `32c967a96b8752c68a8fc00702a8b0acdf4eacffabb4911e9772eb258771629b`
- They differ although integration/ source is unchanged between b6a1290 and e6f8b67 (non-reproducible .NET build); accepted by the coordinator.
- drive.sh installed 12:4xZ, md5 `49b188aa2b060a23efb63777a94e86ae` (previous: `3d915dc30d7444dca544be83b447fbef`, kept as `~/r5/drive.sh.r5e-pre-r5f`). Each batch is pinned to the checkout matching its collection.json learner_commit (b6a1290 -> r5e, e6f8b67 -> r5f); fresh batches use r5f; the launch log line ends with `checkout=`. `fit8-b03` is replaced by `fit8-b03r` (same worlds), also in the post-collection dataset list, which stays on r5e. `~/r5/fit8-b03` and `fit8-b03.FAILED` are untouched evidence.
- Driver restarted 12:46:44Z (pid 44834); the running b02/b04/b05 batches were not interrupted.
