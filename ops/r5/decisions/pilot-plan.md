# r5 critic pilot: partial data, privileged arm, 1 seed (NOT gate evidence)

Reviewed by the co-architect, 27 Oct ~05:00Z. Derived from the harness audit at b6a1290. Steps marked (U) are not yet executed.

## Inputs (GPU VM)

| Item | Path |
|---|---|
| Datasets | gs://anthropocene-rl-datasets-508912/model9/r5/pilot/{fit8-b01,fit8-b03,fit8-b04}/ → `$I/` |
| Actor normalizer, the one the fit8 data were collected against | gs://.../model9/r5/pilot/actor-normalizer.pt → `$NORMALIZER` |
| Code | ~/r5gpu/code-b6a1290/rl |
| Python | ~/r5gpu/venv/bin/python |
| Bridge for `initialize` | the portable bridge already on the disk (~/r5gpu/bridge), with `ANTHROPOCENE_DOTNET` and `ANTHROPOCENE_MODEL9_BRIDGE` set as before |

## Environment

```
export PYTHONPATH=$PWD/src TORCH_DISABLE_NATIVE_JIT=1
cd ~/r5gpu/code-b6a1290/rl
PY=~/r5gpu/venv/bin/python
I=~/r5gpu/pilot
NORMALIZER=$I/actor-normalizer.pt
W=~/r5gpu/pilot-priv-s0
mkdir -p $W
```

## Steps

1. **Provisional privileged normalizer.** Fit on b01 and b03; b04 is passed only as a guard so its worlds are excluded.
   ```
   timeout 1800 $PY -m anthropocene_rl.autonomous_v7.labor_training fit-privileged-normalizer \
     --dataset $I/fit8-b01 --dataset $I/fit8-b03 --calibration-dataset $I/fit8-b04 \
     --report $W/priv-dead-mask.json --out $W/priv-normalizer-b01b03.pt < /dev/null
   ```
   Check that `worlds` covers exactly the b01+b03 worlds.
2. **Initialize.** Needs the bridge.
   ```
   timeout 1800 $PY -m anthropocene_rl.autonomous_v7.labor_training initialize --campaign-r5 \
     --normalizer-checkpoint $NORMALIZER --world-seed 90100 --optimizer-seed 0 \
     --qol-head logistic,td-lambda,0.8 --privileged-normalizer $W/priv-normalizer-b01b03.pt \
     --out $W/checkpoint-000.pt < /dev/null
   ```
3. **Size the schedule.** Run the fit with `--max-steps 1` and read `rows` from the output. Steps per epoch = ceil(rows/256).
4. **Fit in segments**, with `--max-steps` at S, 2S, 3S, …, where S is a multiple of 100 (about 2 epochs' worth). Use the same command each time; each segment resumes from `critic.resume.pt`. Target: 8 epochs, at least 400 steps (one full dominance window) and at least 5 eligible TD refreshes.
   ```
   env TORCH_DISABLE_NATIVE_JIT=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 \
   timeout 10800 $PY -m anthropocene_rl.autonomous_v7.labor_training pretrain-critic \
     --checkpoint $W/checkpoint-000.pt --dataset $I/fit8-b01 --dataset $I/fit8-b03 \
     --require-randomized-tail --epochs 8 --batch-size 256 --lr 3e-4 --max-grad-norm 1.0 \
     --family-weights default,under-clothed=2 --family-weight-decay cosine \
     --stability-log $W/stability-critic.jsonl --stability-every 25 \
     --resume-state $W/critic.resume.pt --resume-every 100 --progress \
     --device cuda --max-steps <S_k> --out $W/critic.pt > $W/fit-<k>.log 2>&1 < /dev/null
   ```
   Watch it live: `grep -E '"kind": "(alarm|rollback|abort)"' $W/stability-critic.jsonl`.
5. **Held-out validation on b04 after each segment.** This runs on CPU.
   ```
   timeout 3600 $PY -m anthropocene_rl.autonomous_v7.labor_pretrain_eval critic-calibration \
     --checkpoint $W/critic.pt --dataset $I/fit8-b04 --fit-dataset $I/fit8-b01 $I/fit8-b03 \
     --batch-size 256 --out $W/cal-b04-seg<k>.json < /dev/null
   ```
6. **Final pilot report.** After the last segment, run the TD residual and the checks snippet from the audit: `td_residual_report` plus `qol_checks`, `population_ranking_checks`, `gap_checks` and `stability_checks`, with the A-dependent checks filtered out. Output goes to `$W/pilot-report.json`.

## Pilot FAIL criteria (approved thresholds)

- StabilityAbort, more than 5 rollbacks, or a gradient-dominance alarm.
- QoL saturation share above 0.05, or above 0.20 in the early stratum.
- TD divergence, a target outside [0, W], or a TD trajectory that is not diminishing.
- A population-tail trigger. Escalate this to the user.
- Privileged attention below 0.10.
- On b04: QoL nMAE above 0.20 or |bias| above 0.10 in any stratum.
- TD residual |bias| above 0.05, MAE above 0.20, or any group above 0.10.
- Population-years ranking below 0.70.
- A gap (b04 versus fit) above 0.10 in nMAE or ranking. This signals overfitting.

## Pilot HOLD criteria

- Fewer than 3 eligible TD refreshes.
- Fewer than 400 steps. The monitor scores an unfilled dominance window as PASS, so check the step count yourself.
- Tiny strata.

## Caveats

The pilot uses 16 fitting worlds and 8 held-out worlds, so the gaps and strata are noisy. It cannot test the A head, determinism, the seed rule or the privileged-vs-plain comparison.
