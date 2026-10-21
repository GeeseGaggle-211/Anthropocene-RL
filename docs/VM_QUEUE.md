# VM job queue (elderq)

A small queue on the training VM that runs multi-phase jobs (training, evaluation, comparison) under a thread budget,
resumes them after Spot stops, and keeps a status record per job. Laptop wrapper: `tools/vmq.ps1`
(`vmq status` == `elderq status` on the VM). The text below is exactly what `elderq help` prints.

```text
elderq - queue of PPO/evaluation jobs on the training VM. Jobs run on their own, in order, and survive VM restarts.

SUBCOMMANDS
  elderq submit <jobfile>          validate the job file and queue it
      elderq submit ~/jobs/r3.yaml
  elderq status [job] [--json]     all jobs (one line each), or one job in detail
      elderq status r3
  elderq logs <job> [phase] [--tail N]   last N lines (default 40) of a phase log; phase 'checkout' is the code checkout;
                                   without a phase, shows the running (or latest) phase
      elderq logs r3 train --tail 100
  elderq cancel <job>              stop a running job or drop a pending one
      elderq cancel r3
  elderq results <job>             list the files under ~/results/<job>/ (add --json for agents)
      elderq results r3
  elderq help                      this text
Add --json to submit, status, results for machine-readable output.

JOB FILE (YAML, or JSON in a .json file)
  name: r3                          # unique; letters, digits, _ - .  Code goes to ~/code/<name>, results to ~/results/<name>
  rl_sha: <40-hex RL commit>        # full SHAs, both already PUSHED to origin
  sim_sha: <40-hex simulation commit>
  threads: 14                       # thread budget; sets OMP_NUM_THREADS; runner cap (default 56) bounds the running total
  priority: 0                       # optional, higher runs first; ties go to the earliest submission
  phases:                           # run in order
    - name: eval                    # unique within the job
      run: python -m pkg.module --out $RESULTS/eval.json   # shell command, run in ~/code/<name>/rl
      done: eval.json               # marker path relative to $RESULTS; the phase is skipped if it already exists
  Each phase gets env vars RESULTS (~/results/<name>), CODE (~/code/<name>, contains rl/ and sim/) and JOB (the name),
  plus OMP_NUM_THREADS. Python is ~/m9-bench/venv/bin/python; use PYTHONPATH=$CODE/rl/src when needed.
  Comments: whole lines starting with #, or ' # text' after a value (not after 'run:', whose command may contain #).
  A multi-line command uses 'run: |' followed by indented lines.

HOW IT BEHAVES
  Jobs start by priority while the running jobs' thread budgets sum to <= the cap; a job that does not fit waits (smaller
  lower-priority jobs do not jump ahead). A phase that exits non-zero, or exits 0 without creating its done marker, marks
  the job failed and stops it; other jobs continue. After a VM restart the runner resumes each unfinished job at its first
  phase whose done marker is missing. States: pending, queued, running, done, failed, cancelled.

RULES
  1. Launch only from pushed commits: both SHAs must exist on origin, because the checkout fetches them.
  2. Never edit a running job: its code checkout (~/code/<name>) is immutable and its job file is copied on start.
     To change anything, submit a new job with a new name.
  3. Every phase must be idempotent: it may be re-run after a restart or crash, so it must resume or overwrite cleanly,
     write its done marker last, and only after its output is complete.

```

## Laptop usage

```powershell
powershell -File toolsmq.ps1 submit C:\path
3.yaml     # copies the file up (relative name, from its folder), then queues it
powershell -File toolsmq.ps1 status --json
powershell -File toolsmq.ps1 logs r3 train --tail 100
```

Defaults: VM `anthropocene-trainer-60`, zone `us-central1-c`, project `anthropocene-508912`; override with
`-Vm`, `-Zone`, `-Project` placed before the subcommand. Commands travel base64-encoded over IAP SSH with strict host keys.

## Scheduling and files (decisions)

- Strict priority: the highest-priority pending job that does not fit under the cap blocks lower ones (no backfill).
- A job whose `threads` exceeds the cap fails immediately with a message. The cap is `--cap` (default 56), overridden live by
  `~/queue/config.json` `{"cap": 48}`.
- A phase must create its done marker; exit 0 without it fails the job. Names are unique across all jobs ever submitted;
  to retry a failed job submit it again under a new name (a new checkout and results dir).
- `~/queue/jobs/<name>/{job.json,status.json,logs/<phase>.log,logs/checkout.log}`; results in `~/results/<name>/`.
  `status.json` has state, times, per-phase state / exit_code / attempts, and the PROVENANCE text.
- Resume: a job left `running` by a stopped VM is picked up again by the next runner, skipping phases whose markers exist.
  The runner uses `KillMode=process`, so restarting the runner alone does not kill phases (they are idempotent anyway).

## Install and deploy (main session, on the VM; nothing here has been installed)

1. Copy the files (from a checkout of branch `feature/r3-queue`, or via `vmq`-style scp):
   ```bash
   mkdir -p ~/queue/bin && cp tools/vmqueue/{elderq,runner.py} ~/queue/bin/ && chmod +x ~/queue/bin/elderq ~/queue/bin/runner.py
   ```
2. Start the runner with systemd (preferred; needs `loginctl enable-linger $USER` once so it survives logout/boot):
   ```bash
   mkdir -p ~/.config/systemd/user && cp tools/vmqueue/elderq-runner.service ~/.config/systemd/user/
   systemctl --user daemon-reload && systemctl --user enable --now elderq-runner && loginctl enable-linger $USER
   ```
   Or cron: `crontab -e` and add the line from `tools/vmqueue/elderq-runner.cron`, then start once by hand with
   `nohup flock -n ~/queue/runner.lock python3 ~/queue/bin/runner.py run >> ~/queue/runner.log 2>&1 &`.
3. Check: `~/queue/bin/elderq status` (prints `no jobs`) and `systemctl --user status elderq-runner`. Optionally
   `ln -s ~/queue/bin/elderq ~/bin/elderq`; `tools/vmq.ps1` calls `$HOME/queue/bin/elderq`.
4. `~/git/checkout.sh <rl-sha> <sim-sha> <name>` must exist and be executable by bash (the runner calls it and requires
   `~/code/<name>/PROVENANCE` afterwards). Existing drivers (t1/t2/r2g) are not affected.
5. Update later by copying the two files again and `systemctl --user restart elderq-runner` (running phases keep going).

## Worked example: the shape of r2g.sh

An evaluation (the five candidate-shard processes run inside one phase and the phase writes a marker last) followed by
compare phases. Replace the SHAs with pushed commits. `$RESULTS` replaces `~/r2g/runs`; inputs still live in `~/r1/inputs`
and `~/r2/runs`.

```yaml
name: r3-grouped-eval
rl_sha: 0123456789abcdef0123456789abcdef01234567
sim_sha: 89abcdef0123456789abcdef0123456789abcdef
threads: 35          # 5 shards x 7 workers
priority: 5
phases:
  - name: eval
    done: eval-all.done
    run: |
      set -u
      PY=~/m9-bench/venv/bin/python; IN=~/r1/inputs; R=$RESULTS
      export ANTHROPOCENE_DOTNET=/usr/bin/dotnet PYTHONPATH=$CODE/rl/src
      REF=~/r2/runs/reference.pt; S55=~/r2/runs/s55.pt
      CANDS=($REF $S55 $(ls ~/r2/runs/*-s6?/latest.pt))
      WORLDS="--default-seeds 2000-2019,2026-2059 --fixture-seeds 2000-2019 --grassland-seeds $IN/grassland-heldout-seeds.txt --grassland-count 10 --valley $IN/valley-relaxed-heldout.txt --valley-strict $IN/valley-strict.txt"
      for k in 0 1 2 3 4; do
        [ -f $R/eval-$k.json ] && continue
        list=(); for i in "${!CANDS[@]}"; do [ $((i % 5)) -eq $k ] && list+=(${CANDS[$i]}); done
        $PY -m anthropocene_rl.autonomous_v7.stage_eval --init $IN/stage1-seed2.pt --candidates $(IFS=,; echo "${list[*]}") $WORLDS           --movement --region-movement --grouped-region --workers 7 --out $R/eval-$k.json > $R/eval-$k.log 2> $R/eval-$k.err &
      done
      wait
      for k in 0 1 2 3 4; do [ -f $R/eval-$k.json ] || exit 1; done
      touch $R/eval-all.done
  - name: compare-cap
    done: compare-cap-vs-reference.json
    run: |
      PY=~/m9-bench/venv/bin/python; R=$RESULTS; REF=~/r2/runs/reference.pt
      export PYTHONPATH=$CODE/rl/src
      $PY -m anthropocene_rl.autonomous_v7.paired_compare --report $R/eval-*.json --a $REF         --b $(ls ~/r2/runs/cap-s*/latest.pt | paste -sd, -) --reference $REF --out $R/compare-cap-vs-reference.json
  - name: compare-all
    done: compare-all-vs-reference.json
    run: |
      PY=~/m9-bench/venv/bin/python; R=$RESULTS; REF=~/r2/runs/reference.pt
      export PYTHONPATH=$CODE/rl/src
      $PY -m anthropocene_rl.autonomous_v7.paired_compare --report $R/eval-*.json --a $REF         --b $(ls ~/r2/runs/all-s*/latest.pt | paste -sd, -) --reference $REF --out $R/compare-all-vs-reference.json
```

A `done` file must appear only after complete output (write to a temporary name and `mv`, or `touch` last), because the
runner trusts it on resume. The per-shard `[ -f ... ] && continue` guards keep the eval phase idempotent after a restart.
