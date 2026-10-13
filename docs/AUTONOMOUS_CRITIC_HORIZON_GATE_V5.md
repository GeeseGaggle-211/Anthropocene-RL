# Autonomous Elder residual critic and physical horizon gate v5

The user authorizes the next safe gate, adds H400 alongside H100, and requires
the independent reviewer to use Astra at high reasoning effort. Preserve all
old worktrees, data, experiments and checkpoints. Rewards remain exclusive to
each controlled village; daughters have separate controllers and contribute no
reward to their parent. No main merge, GPU training or ecology work is included.

## Versioned candidate and value preservation

Learner model/checkpoint/inference v4 retains the v3 actor, action distributions,
761 features and native action3/observation4/environment4/replay4 contracts.
Compare critic LR .0001 with the old critic architecture against a NEW residual
LayerNorm architecture at the same LR. Actor LR remains .0003, legal mixture .05,
reverse-KL .05, reward scale120 and reference lambda .95 per20 years.

For residual-LN, V(s) = frozen_old(s) + (trainable_LN(s) - frozen_initial_LN(s)).
Subtract the two identical normalized outputs BEFORE adding the old value, so
entry values are bitwise preserved for every input. Insert sample-wise LN before
each adaptive critic tanh (epsilon1e-5, initial affine scale1/bias0). Preserve all
old named parameters and populated Adam moments; initialize optimizer state only
for the new LN parameters. Both immutable references are serialized and excluded
from the optimizer. They may contribute input derivatives but never parameter
gradients. Actor, RNG and episode cursors survive explicit v3-to-v4 conversion.
All forward, bootstrap, diagnostics, checkpoint and inference paths use composite
V. Report adaptive health separately from both frozen references and full V.

This is not ordinary LN insertion or evidence that the frozen saturated teacher
has become plastic. The normalized correction supplies a new trainable path.
Three critic forwards and two extra frozen networks add bounded CPU/memory cost;
measure it rather than claiming a free conversion.

## Prospective conversion gate

Use the fixed192-row TRAIN probe and metadata. World831001/831002 provide128
construction rows; whole world831003 provides64 disjoint TRAIN validation rows.
No fitting is needed or allowed for analytical transfer, and no DEV/HELDOUT data
is used for construction. Require exact original values on both splits, unchanged
actor/head parameters and likelihoods, original named Adam moments, RNG/cursors,
frozen anchors excluded from optimizer and stable save/resume/inference.

Across all56 unique original targets from the six selected continuation
checkpoints, fork both candidate variants and use the SAME original value-plus-
label targets. Run64 critic-only updates ×4 minibatches of32; require final MSE
AND maximum last-eight MSE below .9 for every target. Cap112 targets,28,672
critic minibatches,300 seconds, one Torch thread, zero native/action draws and
zero retained fitting. Preserve original bytes and failed/partial evidence.
Require composite value-forward timing less than4.5× the parent, and record
parameter bytes, adaptive derivatives/saturation and frozen-reference health.
Run meaningful populated-Adam, policy-likelihood, H400/truncation and anchor
invariance tests before native launch. Astra/high reviews concrete code/pins.

## New native allowance and H factorial

The prior98,304 additional-training transitions stay recorded as exhausted.
This user-authorized gate has a NEW allowance of32,768 native training transitions:
two critic variants × H100/H400 × seeds29/43 ×4,096 each. Cumulative additional
native training cap becomes131,072. This is not a reset of the previous ledger.
Hard native-campaign wall limit2,400 seconds, four native CPU workers, one Torch
thread,60-second IPC deadlines. Pin original final v3 parents, bank, old native
bridge, preflight and source/config digests before launch. Source and old native
bridge remain unchanged; the heuristic audit uses a separately built executable.
Record read-only evaluation/replay separately: at most6,144 trajectory steps and
6,144 replay resolves at24 boundaries, in addition to32,768 training transitions.
Reserve incomplete collection/replay at its full planned size on interruption;
release only after recording actual completed steps. Maximum native resolves are
45,056 for this campaign. Report actual positive return variance in explained
variance and normalized MSE, using only a1e-12 numerical floor and marking
near-constant returns explicitly.

Keep the eight-decision training bank and all survival/trade/migration/fission
families active. Each turn is20 years. H affects both exp(-dt/H) and the integral
of reward over the interval; it does not change raw welfare or chronology. H400
is a deliberate warm-start target change from an H100-trained teacher, not proof
of400-year policy competence. Report target scale, bias, explained variance and
scale-normalized critic error. Keep lambda and reward scale fixed. Do not reject
legal poor economic outcomes or rescue choices.

Save/evaluate entry and2,048/4,096-transition boundaries. Use the same32 fixed
eight-decision development episodes with replay. Record hidden health, trainable
LN affine growth, value/residual statistics, actual actor/critic parameter steps,
gradients, PPO likelihood/KL, memory and complete collection/update cost. At each
boundary, run a disposable retained-Adam64-update critic test on32 balanced
fixed TRAIN observations; require final and last-eight MSE below .9. Retain the
fresh16-step all-covered-face recovery monitor, clearly labeled fresh-state.
Stop on numeric/replay errors, reproducible adaptation failure, actual optimizer
stall or resource growth; preserve the stopped state and ledger. Never change
thresholds after observing results.

## Final retained actor and native checks

Use four core retained-Adam PPO tasks (workforce, movement, quote enable and field
expansion) on all eight final/stopped candidates. Additionally select, per critic
variant across ALL saved horizons/seeds/boundaries, the checkpoint with greatest
legal nondegenerate categorical confidence and the checkpoint with greatest
single ACTOR-layer saturation. Break ties by cell order then earliest boundary.
On the worst-confidence checkpoint run all13 currently covered retained tasks;
on the worst-actor-saturation checkpoint run the four core tasks. Deduplicate
(checkpoint, task). Maximum66 tasks/135,168 freshly sampled discarded actions,
64 PPO updates ×32 samples/task,900-second cap per checkpoint,4,200-second total
diagnostic wall cap. Preserve the exact-state baseline and actor response/exposure
criteria; require critic final and last-eight errors below .9. Insufficient
coverage and timeout do not pass. This coverage is not all13 retained tasks on
every final model; disclose it in the verdict. All13 fresh tasks are distinct.

Recheck actual selected v4 checkpoint replay, one-versus-four worker collection,
requested probabilities, next PPO update, populated Adam/RNG and inference, with
anchors exact. Final25-decision inference panel uses the same eight development
snapshots/two samplers per candidate (3,200 trajectory steps max plus replay),
600-second wall cap, zero native PPO updates. Longer reset-duration training is a
separate subsequent comparison; this gate changes critic architecture and H only.

The parallel read-only native heuristic audit is governed by its own v1 contract;
its descriptive welfare advantage never becomes a PPO training acceptance veto.
Independent Astra/high final review decides scoped readiness. Learned daughter
lifecycle, reciprocal learned peers and rare livestock/advanced faces remain
explicit integration/coverage limits. A gate pass does not authorize unrestricted
GPU or all-domain scale-out.
