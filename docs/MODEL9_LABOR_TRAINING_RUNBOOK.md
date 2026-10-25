# Model 9 specialty-labor learner: bounded local runbook

This runbook exercises the opt-in specialty-labor learner as a local validation workload. It is
not a readiness comparison, treatment-selection run, or replacement for the reviewed three-arm by
five-optimizer-seed VM design. Confirmation worlds 3000–3059 remain sealed.

The example uses exactly three local worlds:

- world 50000 supplies the physical rollout and the independent training anchor;
- worlds 50001 and 50002 supply held-out actor-gate evidence; and
- all artifacts bind the same initial checkpoint, sampled-policy contract, normalizer, registry,
  native producer, task/cost target transforms, and critic weights used for rollout advantages.

Sharing world 50000 between the rollout and training anchor is allowed because neither is held-out
calibration evidence. The gate worlds must not overlap either source or any world already persisted
in the checkpoint's critic-training history. Thresholds are declared in the report. Do not lower
them after seeing results merely to open the gate; insufficient support leaves the actor closed and
the command performs a critic-only update.

## Prerequisites

Run from the `r3-build-rl` worktree. Supply a bounded-v2 normalizer checkpoint produced under
`model9-bounded-physical-history-v1`; a legacy or mismatched normalizer fails closed.

The currently reviewed alternate native bridge is:

`integration/bin/model9-actor-labor-v1/Anthropocene.Model9Bridge.dll`

Its required SHA-256 is
`6242e35e68e6c63611c30c56430f57047b4837a8de6dc4618bde2d0236ffbfda`.
Verify it before collection; the earlier `819a…` receipt is obsolete.

```powershell
Set-Location C:\Users\suyas\Documents\ChatGPT\worktrees\r3-build-rl
$taskPython = 'C:\Users\suyas\Documents\ChatGPT\worktrees\elder-model9-native-rl\.venv\Scripts\python.exe'
$laborBridge = (Resolve-Path '.\integration\bin\model9-actor-labor-v1\Anthropocene.Model9Bridge.dll').Path
$expectedBridgeSha = '6242e35e68e6c63611c30c56430f57047b4837a8de6dc4618bde2d0236ffbfda'
if ((Get-FileHash -Algorithm SHA256 $laborBridge).Hash.ToLower() -ne $expectedBridgeSha) {
    throw 'Labor bridge SHA-256 differs from the reviewed receipt.'
}
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:ANTHROPOCENE_MODEL9_BRIDGE = $laborBridge
$boundedNormalizer = 'C:\ABSOLUTE\PATH\TO\bounded-normalizer-checkpoint.pt'
$taskOutput = 'C:\ABSOLUTE\PATH\TO\labor-local-validation'
New-Item -ItemType Directory -Force -Path $taskOutput | Out-Null
```

## 1. Initialize the versioned learner

Initialization constructs the dynamic specialty/activity model, two task critics, and two cost
critics from a real labor observation. It saves optimizer, sampling RNG, shuffle RNG, rollout-order,
normalizer, registry, and empty critic-training-history bindings.

```powershell
& $taskPython -m anthropocene_rl.autonomous_v7.labor_training initialize `
  --normalizer-checkpoint $boundedNormalizer `
  --world-seed 50000 `
  --optimizer-seed 4242 `
  --out (Join-Path $taskOutput 'checkpoint-000.pt')
```

Personality conditioning is declared here and nowhere else: add `--personality dirichlet:ALPHA`
(optionally `--personality-key`, `--personality-drift`, `--film-bound`) to sample a personality per
founding lineage; every later phase reads the distribution from the checkpoint. The default
`balanced` reproduces the pre-FiLM model bitwise. See `docs/LABOR_PERSONALITY.md`.

### Task-critic value support (versioned representation)

Task critics fit `y = clip(symlog(G / s), low, high)` two-hot over `bins` values and decode
`s * sum_j p_j symexp(b_j)`. The historical support `-2,9,255` decodes up to about 8102 reference
scales, so stray upper-bin mass with negligible cross-entropy cost (1e-4 in the top bin adds about
0.81 normalized units) can dominate the decoded value (brief section 17.2). The support is part
of the labor model contract: the default is implicit and never serialized (old checkpoints and
contracts are bitwise unchanged); any other support is recorded as `criticValueSupport`
(`labor-task-symlog-two-hot-support-v2`) and every loader constructs the critic from it.
Changing it is a declared representation change, not a tuning knob, and calibration reports are
only comparable at the same support.

* Audit targets first: `python -m anthropocene_rl.autonomous_v7.labor_critic_support audit
  --dataset <fit-data> --out audit.json` reports per-head `symlog(G/s)` quantiles, clipping
  fractions for candidate supports and a recommended support (observed range plus a margin).
* New learner: `initialize ... --critic-value-support=LOW,HIGH,BINS`.
* Campaign r5 (intervention brief 2026-10-26, item 1): new checkpoints use `initialize ...
  --campaign-r5`, which declares `LABOR_CAMPAIGN_R5_VALUE_SUPPORT` = `-0.5,2.5,255`
  (`labor_model.py`). An explicit `--critic-value-support` must agree with it. Without the flag
  the implicit historical default is unchanged.
* Existing checkpoint: `labor_critic_support convert --checkpoint in.pt
  --critic-value-support=LOW,HIGH,BINS --out converted.pt` re-initialises only the two task-head
  output layers (seeded, default the optimizer seed) and keeps the task encoder, cost critics,
  actor, normalizer and trainer RNG; the critic optimizer is reset and the conversion is appended
  to `critic_value_support_conversions`. Continue with `pretrain-critic --checkpoint
  converted.pt`. Tools that splice critic tensors into another checkpoint must also copy the
  contract's `criticValueSupport`.
* `labor_pretrain_eval critic-calibration` reports, per task head, the clip fractions, the
  probability mass and decoded contribution of bins above the observed target maximum
  (`upper_mass`, `upper_contribution`, `upper_excess`, in reference-scale units comparable with
  bias) and the top-bin mass.
* Critic returns discount by `exp(-years / 400)` (`labor_bc.CRITIC_DISCOUNT_YEARS`) using the
  per-row `transition_elapsed_years` (20 per native turn). Shards written before 2026-10-26 store
  the same year values under the misnamed `transition_elapsed_days`; `load_labor_dataset` renames
  it without conversion. Critic summaries report `discount_years` (formerly `discount_days`).
* `pretrain-critic` accepts only heuristic-teacher shards (`native_heuristic_labor_teacher`,
  `row_source` 0). It refuses search shards (`native_search_labor_teacher`, `row_source` 2),
  learner-state DAgger/actor-only shards and multi-source aggregates
  (`labor_bc.require_critic_teacher_dataset`).
* `pretrain-critic --cost-loss-weight W` (dataset path only) scales the summed cost MSE in the
  joint, globally clipped critic loss; the default 1.0 is unchanged and not recorded.

## 2. Collect one frozen-behavior physical rollout

Collection uses the typed two-proposal bridge. An initial proposal scores labor plus retained
movement/fission/trade; a retry, if requested by typed validation, resamples and scores labor only.
The simulator advances exactly once per physical parent decision. Every living sequence tail must
have a complete finite four-head bootstrap; only a true terminal tail may use zero.

```powershell
& $taskPython -m anthropocene_rl.autonomous_v7.labor_training collect `
  --checkpoint (Join-Path $taskOutput 'checkpoint-000.pt') `
  --worlds 50000 `
  --turns 25 `
  --out (Join-Path $taskOutput 'rollout-000.pt')
```

## 3. Collect the independent training anchor

This uses native forks, four sampled continuation paths, a fixed 25-step prefix, and prespecified
geometric randomized-tail stopping draws. It does not use a critic bootstrap or treat truncation as
termination. World 50000 remains training data, not held-out calibration evidence.

```powershell
& $taskPython -m anthropocene_rl.autonomous_v7.labor_anchor_producer `
  --checkpoint (Join-Path $taskOutput 'checkpoint-000.pt') `
  --worlds 50000 `
  --optimizer-seed 4242 `
  --continuation-seed-base 60000 `
  --role train-anchor `
  --out (Join-Path $taskOutput 'train-anchor-000.npz')
```

## 4. Collect held-out actor-gate evidence

The held-out source uses worlds 50001 and 50002. Its ranking witnesses compare distinct successor
states using immediate reward plus discounted successor value against independently realized branch
returns. They never ask a state-value critic to rank alternative actions at the same parent state.

```powershell
& $taskPython -m anthropocene_rl.autonomous_v7.labor_anchor_producer `
  --checkpoint (Join-Path $taskOutput 'checkpoint-000.pt') `
  --worlds 50001,50002 `
  --optimizer-seed 4242 `
  --continuation-seed-base 70000 `
  --role actor-gate `
  --out (Join-Path $taskOutput 'actor-gate-000.npz')
```

## 5. Emit the current four-head calibration report

The calibration command loads the exact checkpoint and binds the effective sampled actor, critic
weights, bounded normalizer, registry, held-out file digest, world IDs, independent-unit IDs, and
declared thresholds. The JSON always states `localReadinessClaim: false`.

```powershell
& $taskPython -m anthropocene_rl.autonomous_v7.labor_value_calibration `
  --checkpoint (Join-Path $taskOutput 'checkpoint-000.pt') `
  --anchor (Join-Path $taskOutput 'actor-gate-000.npz') `
  --out (Join-Path $taskOutput 'calibration-000.json')
Get-Content -Raw (Join-Path $taskOutput 'calibration-000.json')
```

An open gate requires every task/cost head to pass coverage, magnitude, tie-aware zero-target, and
applicable distinct-successor ranking checks under the declared thresholds. Two held-out worlds do
not guarantee that support. A closed gate is an expected, valid outcome and must not be reinterpreted
as readiness.

## 6. Run one frozen-behavior update

The command rejects an off-policy training anchor, stale rollout, normalizer mismatch, stale critic
binding, or overlap between gate worlds/units and current or historical critic-training data. Critic
attempt losses use the physical-parent denominator; the independent anchor is separately weighted.
If the report is valid but closed, only critics update. If open, actor minibatches use one permutation
per epoch and post-step fixed-audit KL rollback restores both parameters and Adam state.

```powershell
& $taskPython -m anthropocene_rl.autonomous_v7.labor_training train `
  --checkpoint (Join-Path $taskOutput 'checkpoint-000.pt') `
  --rollout (Join-Path $taskOutput 'rollout-000.pt') `
  --anchor (Join-Path $taskOutput 'train-anchor-000.npz') `
  --calibration-report (Join-Path $taskOutput 'calibration-000.json') `
  --out (Join-Path $taskOutput 'checkpoint-001.pt')
```

The output checkpoint's `last_update` records critic losses, per-head independent-anchor losses,
actor-gate status, attempted/accepted actor steps, rollback status, and exact epoch coverage. Its
critic-training world/unit history prevents later held-out reuse. A new rollout is required for the
next update. Re-run calibration against the new checkpoint; if the actor changed, independent return
data bound to the old policy is stale and must be refreshed before another actor update.

## Scope and acceptance status

This path is source- and unit-tested and has one reviewer-controlled native collector receipt:
world 50000, policy seed 4242, two turns, 16 physical parent rows, no sampled retries, nine living
bootstrap camps, finite four-head targets/losses, and zero optimizer steps. That receipt validates
the typed runtime path; it is not a training result. No local output may select an arm or establish
readiness. Readiness-bearing comparisons still require at least three matched arms and five paired
optimizer seeds per arm on the training VM, unless a different design is reviewed in advance.
