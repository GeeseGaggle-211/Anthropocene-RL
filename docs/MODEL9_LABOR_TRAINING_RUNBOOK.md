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

Bridges built from simulator `feature/r5-privileged` emit economic accounting schema 3: herd rows
add `acquiredAnimals` (taming) and `grantedAnimals` (opening compatibility grant) and a real fission
`ownershipInflow`, with `herd-acquisition` and `herd-compatibility-grant` events and transfer
kinds. The learner reads schemas 2 and 3; for schema 3 the herd conservation row binds
production = births + grants and transferIn = acquisition + ownership inflow
(`economic_accounting.py`). Build that bridge with
`dotnet build -c Release integration/Anthropocene.Model9Bridge.csproj
-p:AnthropoceneSimRoot=<r5-sim checkout> -o integration/bin/model9`.

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
* Critic-target tail (intervention brief 2026-10-26, item 1b). `labor_teacher_data` now closes
  every living (world, camp) cutoff at the collection horizon with the anchors' randomized tail
  (`--return-tail randomized`, the default; preregistered `--tail-draws 16 --tail-rho 0.95
  --tail-cap 200`, prefix = the 25-decision window). After the window the converted teacher
  keeps playing without recording rows for the frozen schedule's `required_terms` (the largest
  of 16 geometric stops; per-world schedule seed `--tail-schedule-seed-base` + world seed), and
  `labor_anchor_producer.summarize_labor_path` with `prefix_terms=0` gives each cutoff camp an
  exact terminal value or a Russian-roulette estimate valued at the horizon state. Each row
  stores its trajectory's typed target (`tail_method`, `tail_cutoff_turn`,
  `tail_value__<head>`, `tail_standard_error__<head>`); the shard meta stores
  `return_tail_record` with the schedule, per-camp entries and the continuation cost.
  `discounted_returns` starts each living trajectory from its tail, refuses a living cutoff
  without one, and `critic_target_available` excludes `unavailable` tails (hard cap, or a camp
  that left the live set without terminating) from fitting and calibration. The SE is
  stopping-randomization only: teacher and simulator are deterministic. `--return-tail
  truncated-zero` collects the legacy window, which shards then declare; `pretrain-critic
  --require-randomized-tail` refuses such data. Critic summaries and calibration reports name
  the method (`return_tail`).
* `pretrain-critic --cost-loss-weight W` (dataset path only) scales the summed cost MSE in the
  joint, globally clipped critic loss; the default 1.0 is unchanged and not recorded.

### Critic-only privileged state (intervention A, schema 1)

The three critic encoders (task, feasibility, settlement guard) can read a versioned, critic-only
view of hidden economic state; the actor stays observation-only. The simulator
(`feature/r5-privileged` 8bd480b, `ElderModel9Privileged`) emits it as `privilegedView` with
`privilegedSchemaVersion` on every village row, captured at the same decision boundary as the
observation (row schema 1: reward/outcome fields belong to the executed transition, observation
and privileged view to the next decision). The bridge advertises the layout in its schema op.

| Family | Rows (cap + overflow) | Width | Tokens |
|---|---|---|---|
| `core` | 1 | 10 | 1 (the earlier ten floats, bit-identical) |
| `fields` | 16 + 1 | 19 | 17 |
| `herds` | 8 + 1 | 24 | 9 |
| `unlocks` | 1 + 1 | 28 | 2 |
| `economy` | 1 + 1 | 64 | 2 |

Feature names are pinned in `labor_privileged.LABOR_PRIVILEGED_LAYOUT_V1` and checked against the
live bridge. Each family has its own shared row projector plus a family embedding (and an
overflow-row embedding on its last row) into the critics' global attention set: 90 observation
tokens plus 31 privileged tokens, masked by presence, under the same global FiLM. Per encoder
this adds 20,352 parameters (task 796,160 to 816,512; each cost encoder 729,344 to 749,696); the
actor is unchanged (864,912).

Isolation, enforced in code and tests (`tests/test_labor_privileged_*.py`):

* `bridge.observation_batch` never includes `privileged`/`privilegedView`;
  `bridge.privileged_batch` is the separate critic-only batcher. Normalized critic inputs
  (`privileged__<family>`, `privileged__<family>_present`) are passed to a critic as its own
  argument (`task(tensors, privileged)`), never merged into the actor batch.
* `LaborActor.forward` and every encoder assert that no `privileged*` key is in the observation
  batch; `LaborObservationEncoder` rejects a nonzero privileged width in its constructor, and
  `LaborActor` receives `layout.actor_layout()`. Only `LaborCriticEncoder` accepts the block.
* Actor outputs are bitwise invariant under privilege-only perturbation; critic outputs change.

Workflow (fitting worlds only for the moments; never 3000-3059, 2000-2059, the reserved
90154-90159 or calibration worlds 90180-90199):

```powershell
# 1. Teacher shards now store raw privileged_<family>, privileged_<family>_present and
#    privileged_schema_version (also DAgger and search rows).
& $taskPython -m anthropocene_rl.autonomous_v7.labor_teacher_data --normalizer-checkpoint $boundedNormalizer `
  --worlds 90100-90153,90160-90179 --turns 25 --out $fitData
# 2. Frozen per-family moments plus the dead-mask report; audit it before any training.
& $taskPython -m anthropocene_rl.autonomous_v7.labor_training fit-privileged-normalizer `
  --dataset $fitData --calibration-dataset $calData --report dead-mask.json --out privileged-normalizer.pt
# 3. New learner with privileged critics (actor unchanged).
& $taskPython -m anthropocene_rl.autonomous_v7.labor_training initialize --campaign-r5 `
  --normalizer-checkpoint $boundedNormalizer --privileged-normalizer privileged-normalizer.pt `
  --world-seed 90100 --optimizer-seed 4242 --out checkpoint-000.pt
& $taskPython -m anthropocene_rl.autonomous_v7.labor_training pretrain-critic --checkpoint checkpoint-000.pt `
  --dataset $fitData --epochs 4 --batch-size 256 --lr 3e-4 --max-grad-norm 1.0 --out critic.pt
& $taskPython -m anthropocene_rl.autonomous_v7.labor_pretrain_eval critic-calibration --checkpoint critic.pt `
  --dataset $calData --fit-dataset $fitData --out calibration.json
```

* The privileged normalizer is schema v2 (`labor-privileged-normalizer-v2`): each feature has a declared
  transform (`labor_privileged.PRIVILEGED_TRANSFORMS`, symlog by default; `per-year-log1p:0.1` for the
  per-day regrowth rate; `symlog>identity` -- frozen at fit time -- for large near-constant stocks and
  capacities and near-one nutrition fractions; fixed `count-log1p` for the structural overflow row-count),
  then a frozen affine (std floor 0.1 except raw z-scores, guard 10). A field is dead (zeroed) only when it is
  constant within float tolerance after its transform (std <= 1e-6 max(1, |mean|)). v1 normalizers still
  load and normalize exactly as before. `fit-privileged-normalizer` writes `dead_mask_report` and the data
  audit (`privileged-dead-mask-audit` standalone), which gives a reason for every dead field
  (`constant-in-data`, `varies-only-at-float-rounding`, `never-present`), the legacy v1 flag and the fields
  v2 rescues. Review it before training.
* The actor normalizer keeps its legacy unversioned 10-wide `privileged_mean/std/dead` moments
  byte-identical (teacher shards bind its digest); they are unused. The critics read only the
  schema-1 per-family moments stored at checkpoint key `privileged_normalizer`.
* Checkpoints: `labor_encoder_layout.privileged` persists rows, widths, names and the schema
  version; `labor_model_contract.privilegedCritic` records the architecture
  (`labor-critic-privileged-family-tokens-v1`), encoder widths, layout and normalizer digests;
  `labor_critic_digest` adds the privileged schema version and normalizer digest. A mismatched
  schema version, digest, layout or missing part is rejected at load. Checkpoints without a
  privileged block keep their bytes and load bit-identically.
* Migration: `labor_critic_support convert-privileged --checkpoint in.pt --privileged-normalizer
  privileged-normalizer.pt [--seed N] --out out.pt` loads every existing tensor strictly and
  initializes only the privileged projectors and embeddings (seeded) through the critic encoders'
  strict-load migration hook; the critic optimizer is reset and the migration is recorded in
  `critic_privileged_conversions`.
* Data: `load_labor_dataset(..., require_privileged=True)` (pretrain-critic and critic-calibration
  of a privileged critic) rejects any shard without privileged arrays; they are never
  zero-filled. Actor BC still loads such shards; a partly privileged aggregate drops the arrays
  for the actor. `pretrain-critic` also refuses reserved 90154-90159 and calibration 90180-90199
  rows, and `critic-calibration` refuses a dataset whose worlds fitted the privileged moments.
* Not wired: PPO `collect`/`train` and the anchor-based `labor_value_calibration` call the critics
  without a privileged batch, so a privileged critic fails closed there. The brief authorizes no
  critic-driven actor update.

### Reserved worlds, learner-state V targets (r5 data)

* **Reserved worlds 90154-90159** never train any network (`labor_worlds`, one constant and guard). Teacher,
  DAgger and search collection, PPO rollouts, `pretrain-bc`, `pretrain-critic` and every training dataset
  loader refuse them; `--drop-reserved-worlds` on `pretrain-bc`/`pretrain-critic` instead drops those
  worlds' shards from an older dataset (recorded as `dropped_reserved_worlds`). Teacher and search
  collection run on them only with `--held-out-test` (reserved worlds only; outputs tagged
  `partition: held-out-test`, refused by every training loader; evaluation readers use `training=False`).
* **V^rho targets at learner-visited states.** `labor_dagger collect --teacher-continuation branch-end`
  (or `all-visited`) `--continuation-out DIR` forks each learner-visited state, runs one H12 teacher
  continuation for every live camp with the preregistered randomized tail (`--tail-draws/--tail-rho/
  --tail-cap`, defaults 16/0.95/200) and restores the fork, so the actor rows are byte-identical. The
  continuation dataset (source kind `learner_state_teacher_continuation`, `row_source` 3, one
  `row_trajectory` per continuation, typed `tail_*` fields) is accepted only by
  `pretrain-critic --learner-state-continuations`, alone or with teacher shards
  (`--dataset teacher --dataset continuation`); actor BC refuses it. Cost: one continuation is
  2 + 2 x required_terms resolves (the world's frozen schedule, e.g. 62 terms for world 90100). Measured
  locally (26 Oct, world 90100, 16 players, default map, depth 1, 2 starts, horizon 3, branch-end): 126
  resolves and 3,611-4,585 s per continuation (about 29-36 s per resolve with a concurrent job), against
  140 s for the learner rows themselves, so continuations dominate DAgger cost by about 60x per branch.

### Curated Neolithic scenarios (r5 gap filling, manifest section 11)

`labor_scenarios` fills family-coverage gaps in the 50-decision banks; it is not a parallel bank.

* `coverage --dataset FIT50 --dataset CAL50 --out coverage.json` tags every teacher row with the
  families below and reports rows, roots (run entries: the decision a camp enters a family) and worlds
  per family and partition against the minima (fitting >= 32 roots over >= 4 worlds, calibration and
  held-out >= 16 over >= 4), with entry subtypes (`reset`/`in-trajectory` head windows,
  `initial`/`relapse` housing and clothing) and each food head's first active use (first valid label
  staffing it) and the share of those inside the head's unlock window.
* Families (all require `era-neolithic`; a row may carry several): `farming|herding|fishing-unlocked`
  (first 2 decisions with Farmer/Herder/Fisher legal after an observed unmask, or from the episode's
  first decision for reset camps; daughters inheriting a head open no window), `mixed` (same rule on two
  or more of those heads), `small-camp` (< 100 residents), `large-camp` (> 600), `under-housed` /
  `under-clothed` (privileged `housing-per-capita` / `cloth-per-capita` < 0.3), `fission-daughter`
  (camp absent at the first decision, its first 2 decisions), `trade-eligible` (privileged
  `trade-partners` >= 1, the partners `Trade.Plan` would pair this turn),
  `first-use-farming|herding|fishing` (the first decision whose valid, accepted teacher label staffs the
  head, and the next one; once per camp). Under-housed/under-clothed use hysteresis: enter below 0.3,
  exit above 0.4, and a re-entry is a new root only after two decisions above 0.4 (on the local 90100
  bank this cut under-clothed roots from 1,091 to 330). Family bits are append-only.
* Weight, not support (decision 26 Oct): the 50-decision banks meet the root minima, so the default
  remedy is `--family-weights` on `pretrain-bc`, `pretrain-critic` and `labor_r5_acceptance run`
  (`FAMILY_WEIGHTS=` in `tools/r5_critic_acceptance.sh`). Rows are tagged at load time (teacher
  `row_source` 0 by the tracker, scenario rows by their stored tags, other sources untagged), each row
  takes the max weight of its families (1 untagged), and every (source, world) group is rescaled to its
  unweighted mass. `default` = the weights (proposed, approved 26 Oct with the decay below): first-use x10, legal-window heads and mixed x5,
  under-housed/under-clothed x3, fission-daughter x2, others x1 (`default,family=w` overrides).
  Summaries record the weights, status and before/after shares; `labor_scenarios weights --dataset D`
  (or `--shard`) prints them.
* Decay (user condition, 26 Oct): `--family-weight-decay {none,linear,cosine}` (cosine by default with
  weights) and `--family-weight-decay-epochs E` (default: the continued schedule's, else the fit's
  `--epochs`). Each weight acts as 1 + (w - 1) s(t), s from 1 to 0 over E epochs of the global optimizer
  step: BC rescales each minibatch's `row_weight`; the critic maps one per-epoch uniform per draw
  through that step's CDF (exact resume; the resume fingerprint binds the schedule). Checkpoints carry
  `family_weight_schedules.{actor,critic}` with `epochs_end`; the next pretrain-bc/pretrain-critic with
  the same weights and decay continues from it (DAgger refits never restart at full weight;
  `--family-weight-restart` overrides). The stability JSONL logs `family_weight_factor` and
  `family_weight_multipliers`. Decay `none` keeps the fixed loader weights.
* `food-trade` guard family (never up-weighted): last turn's imported food (observation trade row
  `food`, column `imported`) over the camp's meals per turn (privileged economy `meals-per-turn`)
  above 0.2. Food produced is not observable at the boundary, so meal need is the denominator. Stored
  rows recover imported food by inverting the bounded normalizer, so tagging needs the dataset's
  normalizer and is unavailable when that column is dead in it (the local test normalizer, fitted at
  reset, zeroes it). The acceptance evidence reports per-family V calibration, A ranking (reserved
  roots get their state families from their tensors) and actor fidelity under `by_scenario_family`,
  and the receipt carries the non-gating `advisory:food-trade-vs-all` (FLAG when food-trade nMAE is
  above, or ranking below, the all-rows value by more than 0.05 with weights on).
* `collect --partition fitting|calibration|held-out-test --coverage coverage.json --exclude-dataset BANK`
  drives each world under H12 (`--decisions`, default 50; the VM driver uses 100), keeps a keyed
  bottom-k quota per family and world (caps from the audit's deficits, 0 for met families; or
  `--quota-per-world`), holds a native fork per boundary that still owns a selected root, never roots a
  state a bank with the same reset options recorded, then records the root camps' teacher trajectory for
  `--scenario-decisions` (25) with the randomized tail. Rows: teacher shard format plus `row_source` 4
  (`curated_neolithic_scenario`), `row_scenario_family` (root families bitmask), `row_scenario_tags`,
  `row_root_turn`, `row_start`, `row_start_index`, `row_branch_step`, `row_trajectory` = 1,000,000 +
  root index. Shard metas list roots; `roots.json` feeds `labor_search --source scenario --scenario-roots`
  (payload roots keep source `h12`, root ids end `/scenario`, `scenario_family` is carried into
  `labor_advantage` roots). `--resume` keeps verified shards; a published output is a no-op.
* Training: `pretrain-bc --dataset FIT --dataset SCEN` (equal source mass); `pretrain-critic --dataset FIT
  --dataset SCEN --curated-scenarios` (sampling switches to equal source mass only when scenario rows are
  present). Calibration-partition sets are refused by every training loader; `labor_r5_acceptance run
  --scenario-calibration-data CALSCEN` (or `SCENARIO_CAL_DATA=` in `tools/r5_critic_acceptance.sh`,
  `SCENARIOS=1` for fitting sets in `FIT_DATA`) adds report-only per-family calibration;
  `critic-calibration` reports `by_scenario_family` for any dataset and `labor_advantage_eval` reports
  `by_scenario_family` A ranking when payload roots carry families (reserved roots via
  `--partition held-out-test --held-out-test`).
* VM: `NORMALIZER=... FIT_BANK=... CAL_BANK=... tools/r5_scenarios.sh` (coverage, gap collection per
  partition, cost and post-fill coverage, optional `SEARCH=1`); rerun after a preemption.

### r5 critic acceptance harness (enforced stability and fitting acceptance)

`labor_stability` wraps every optimizer step of `pretrain-critic` (V), `labor_advantage fit` (A) and
`pretrain-bc` (actor BC refits). It is on by default for r5 checkpoints (a privileged block or the
bounded QoL head) and always on for A; `--no-stability` turns it off, `--stability-log PATH` writes the
JSONL and `--stability-every N` sets the logging interval (25). A monitored fit without failures is
bitwise identical to an unmonitored one (tested). The receipt is stored in the fit summary under
`stability` and its `admitted` flag is false when any mandatory check fails or holds.

* Finite checks every step (loss, global pre-clip gradient norm, parameters, Adam moments). On a
  nonfinite value: parameter and Adam rollback to the last good step, LR x 0.5, an event in the receipt;
  K = 3 consecutive failures raise `StabilityAbort` and write `<out>.stability-abort.json`. More than 5
  rollbacks in one fit fail `finite_updates`.
* Every N steps: per-head losses, per-block pre-clip gradient norms (frozen actor/critic blocks must be
  exactly zero and keep their digest), the global clip multiplier and per-block parameter-step norms.
  Dominance detectors: QoL vs population-years on the shared task encoder, task vs each cost on the
  global clip, QoL vs population-years on A's shared action encoder. Alarm: one contender above 0.90 of
  the block's gradient norm on 75% of a 16-point window (16 log points x 25 steps = 400 optimizer
  steps; user decision 26 Oct) while another contender's loss improves by less than 1% (mean of the
  window's second half against its first half).
* Bounded QoL saturation (share of pre-activations with |z| > 6, judged on the final full pass,
  stratified; approved relaxation 26 Oct): low-side rows whose bounded target G/W is below 0.05 are
  expected (no housing/clothing yet) and excluded; the first 5 decisions of each (world, camp,
  trajectory) fail above 0.20 of unexpected saturation, every other row above 0.05.
* TD(lambda): per refresh, drift (mean |target change|) and error (mean |prediction - TD target|) in
  units of W. The first refresh (replacing the Monte Carlo return) is recorded but not counted, and the
  next 2 refreshes are warm-up. Live alarm: 3 consecutive rises each above 10% (and above the 1e-3
  floor), or any target outside [0, W]. End of run: each post-warm-up series must diminish (negative
  log slope and final value at most 0.5 x its maximum, or converged below the floor); too few
  refreshes HOLD.
* Live early stop (`--stability-early-stop`, on `pretrain-critic`, `labor_advantage fit` and
  `pretrain-bc`; off by default, on in `labor_r5_acceptance run`): a gradient_dominance alarm, a live
  td_divergence alarm or more than 5 total rollbacks raises `StabilityAbort` at once. The thresholds
  and definitions are those of the end-of-run checks, so the abort receipt
  (`<out>.stability-abort.json`) FAILs the same check id and adds `early_stop` (check, reason, step).
  The acceptance row carries `stopped_early_at_step`. QoL saturation and the population tail stay
  end-of-run judgements; with the option on, the population tail adds a live non-gating `kind=alarm`
  (minibatch, on entry; listed under `advisories`). Without a trip the fit is bitwise unchanged.
* Held-out validation of V (`pretrain-critic --validation-dataset DIR...`, monitoring only): at every
  epoch end the same loss components (per head, task, cost, total) are computed on the admitted rows
  of those datasets. The pass runs without gradient, in a fixed row order, under a forked and restored
  torch RNG; the trainer generator is untouched. The critics have no dropout or batch norm (refused
  otherwise), so the forward is the fitting graph. Records are `kind=validation`. A TD(lambda) QoL
  target is bootstrapped through the current head on the validation trajectories, and the Monte Carlo
  QoL loss is also reported. The non-gating `validation_divergence` alarm fires when the validation total
  rises on 2 consecutive epochs while the epoch training loss falls (mean of its accepted minibatch
  losses). The validation worlds must be disjoint from the training worlds; reserved and sealed worlds
  are refused, and the training loader still refuses calibration worlds. The fitted parameters are
  bitwise identical with validation on or off (tested on CPU). The result goes in the fit summary and
  the stability receipt under `validation`, and in the acceptance receipt as the advisory
  `advisory:critic-validation-divergence`.
* Population head: decoded upper excess above 0.01 reference scales or top-bin mass above 1e-3 on the
  final pass fails. A: train vs held-out (within-fitting-world split) QoL pair ranking every epoch; a
  final gap above 0.10 or a held-out fall of 0.03 from its best fails. Privileged critics: village-token
  (or any-query) attention mass on privileged keys must reach 0.10 of the uniform share; dead privileged
  columns and dead trunk units are reported (advisory).
* `pretrain-critic --resume-state PATH --resume-every N` writes an exact mid-fit resume state (models,
  Adam, epoch-start RNG, position, TD targets, monitor history) atomically and continues from it
  bit-for-bit; `--max-steps` stops early for probes; `--progress` prints g0launch-style progress lines.

`labor_r5_acceptance` turns the evidence into `gate-receipt.json` (schema
`labor-r5-critic-acceptance-receipt-v1`) plus `evidence-manifest.json`:

```bash
# VM: 2 arms x 5 paired seeds, 4 fits at 14 threads, resumable (rerun the same command).
NORMALIZER=... PRIV_NORMALIZER=... FIT_DATA=... CAL_DATA=... FIT_PAYLOADS=... \
RESERVED_PAYLOADS=... CONTROL_REPORT=critic3-calibration.json OUT=... ATTEMPT_ID=r5-accept-1 \
  tools/r5_critic_acceptance.sh
# Pieces: evaluate one fit, aggregate, compare two checkpoints.
python -m anthropocene_rl.autonomous_v7.labor_r5_acceptance evaluate --arm A --seed 0 --checkpoint critic.pt \
  --advantage advantage.pt --calibration-data CAL --fit-data FIT --reserved-payloads RES \
  --control-report critic3.json --out evidence.json
python -m anthropocene_rl.autonomous_v7.labor_r5_acceptance receipt --evidence */seed-*/evidence.json \
  --seeds 0,1,2,3,4 --seed-rule all-seeds --determinism determinism.json --attempt-id ID --out gate-receipt.json
```

**Catching failures before the scale-out.** `labor_r5_acceptance run` (and so
`tools/r5_critic_acceptance.sh`) passes `--stability-early-stop` to every V and A fit and
`--validation-dataset $CAL_DATA` to every V fit; the calibration directories may not also be fitting
directories. The seed gate runs the first seed of every arm first, concurrently up to `JOBS`. If the
candidate arm's first seed ends in `StabilityAbort` (early stop included) or any stability FAIL, the
remaining seeds are not launched: `seed-gate.json` records the evidence, and the receipt FAILs with a
top-level `seed_gate` and a `seed-gate` unresolved finding. Otherwise the remaining seeds launch as
before. `--no-seed-gate` restores the single launch of every arm x seed;
`--no-stability-early-stop` and `--no-critic-validation` turn the other two off (script:
`SEED_GATE=0`, `EARLY_STOP=0`, `CRITIC_VALIDATION=0`).

**Collection batches.** `FIT_DATA` and `CAL_DATA` (and `FIT_PAYLOADS`, `RESERVED_PAYLOADS`)
are space- or colon-separated lists, e.g. `FIT_DATA="$I/fit8-b01:$I/fit8-b02:...:$I/fit8-b08"`
and `CAL_DATA="$I/cal8-b01 $I/cal8-b02 $I/cal8-b03"`. `pretrain-critic --dataset` (repeatable),
`evaluate --calibration-data/--fit-data` and `labor_pretrain_eval critic-calibration
--dataset/--fit-dataset` take several directories. Several heuristic-teacher batches are read
as ONE teacher dataset (`labor_teacher_data.merge_labor_datasets`). Their shards are pooled in
the world order one published dataset stores them, with equal mass per world over the combined
set. Batches that partition a set of worlds therefore give the same tensors and the same critic
fit, bit for bit, as one dataset over those worlds. Listing order does not matter; only the
provenance digest differs.

Each batch is verified on its own: digests, the reserved-world guard, and the
calibration/held-out-test refusal for training. Across batches, the merge refuses a repeated
batch, a world in two batches, and any difference in normalizer, source kind, return tail or
randomized-tail settings, turns, reset options, partition or stored arrays (privileged layout
included). Critic fits still refuse reserved and G0 calibration worlds. One directory is read
exactly as before. Learner-state continuations and curated scenarios keep their own source mass
beside the merged teacher set. Actor BC (`pretrain-bc`) keeps its per-directory aggregation,
unchanged.

The primary gate `r5-critic-acceptance` (candidate arm) combines: admitted stability receipts for V and
A; QoL per r5-b (a) A same-village ranking on reserved worlds, (b) bounded-head nMAE <= 0.20 and
|bias| <= 0.10 per stratum, (c) TD residual |bias| <= 0.05 and mean |residual| <= 0.20 reference scales;
population-years cross-village ranking >= 0.70 per stratum (r5-b item 3); the other heads within
0.02 of the critic3 control (`--control-report`, a missing control HOLDs); the
A gates on reserved worlds; generalisation gaps; the declared seed rule (`all-seeds` default, or
`mean-with-floor`: seed mean passes and the worst seed is within 0.02, stability and controls in every
seed); the privileged-vs-control comparability (same data digest, support, tail, epochs, batch, lr,
clip, QoL head and A config, else HOLD); determinism (bitwise, or within 1e-6 relative, else FAIL); and
the historical fitting exclusions. Full G2C for every head is reported as the secondary gate
`G2C-all-heads`. Missing evidence HOLDs, a demonstrated violation FAILs, a stability abort is a FAIL,
and `--smoke` (truncated-zero targets) forces HOLD. `--local` caps seeds at 3, so a local receipt HOLDs
on seed coverage unless `--required-seeds` is lowered for a diagnostic. Values marked `proposed` in `PREREGISTERED` await
the user's review.

`labor_qol_diagnosis` (feature/qol-diagnosis, 770ca04) predates the privileged block and the bounded
head (it calls the critic without a privileged batch and decodes a two-hot softmax), so it runs only on
the critic3 control; the harness's calibration strata and TD residual replace it for r5 critics.

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
