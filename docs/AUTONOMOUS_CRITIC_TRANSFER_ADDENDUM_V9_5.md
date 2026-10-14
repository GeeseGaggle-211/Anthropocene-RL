# Frozen TRAIN-derived critic initialization v9.5

Preserve v9.2/v9.3/v9.4 strict failures, data, source/ledger pins and original
actor307targets. The single TRAIN-CV ridge attempt failed; no v9 PPO/DEV ran.
Next use one explicit critic initialization transfer from canonical lowest
passing donor101 to307. Donor weights were fitted on TRAIN; donor eligibility
uses the reused calibration gate. Independent actor initializations remain;
critic307will share donor101initialization and is not an independent critic.

The canonical donor/frozen-transfer option was declared before its read-only
source audit. That audit inspected both fitting and reused validation tensors,
with no weights or hyperparameters changed; it cannot be called untouched new
validation. Freeze this single transfer, with no subsequent32epoch adaptation,
L2 tuning, donor search or validation threshold change. DEV/HELDOUT remains
unused for fitting/selection. Source audit0calls/0updates/.999762seconds chains
to22417cumulative resolves/746.3559672170086activeelapsedseconds.

Keep307full-TRAIN normalization offset/scale exactly. Copy donor live/reference/
initial-normalized critic tensors, multiply each output head byscale101/scale307,
and add(offset101-offset307)/scale307only to live output bias. This algebraically
reproduces donor raw values under existing model7 residual value semantics.
Require measured float32 raw-value parity<=1e-4 on fitting and reused gate inputs.
Explicitly replace and refreeze critic anchors; this supersedes their previous
initialization for this new fork, not the immutable failed checkpoints/targets.
Clear only live critic Adam, preserve actor/heads/populated actor Adam/RNG/counters/
config/value buffers exactly. Unit tests verify likelihood equality, donor
nonmutation, anchor freeze, value parity and protected actor state.

No new physical target or target-generator reinterpretation. Keep original307
signed frozen tails unchanged and record original target provenance, donor and
receiver checkpoint/config hashes and measured affine parity. Successful101/211
model and optimizer tensors stay unchanged. After transfer require all three
unchanged strict gates:EV>=.60,MSE/Var(return)<=.40,absrawbias<=2,and nonworse
original unfit critic. Gate reuse is explicitly calibration, not held-out proof.

Conditional downstream remains six448-transition LR cells (2688retained),
three baselines and six finals over6DEVstarts×25turns,1350+1350replay. At22417
prior resolves,5388maximum→27805cumulative,186reserve under27991cap. Fixed25
turns/H400/actorLR.0003/.00003, four native workers and one Torch thread; actor
probability/control/mask/native physical/reward contracts unchanged. Chain actual
elapsed time and retain2369.4072460189927seconds ceiling. Freeze concrete source,
config,data,binary pins and Astra/high preflight before transfer/PPO; independently
review actual updates/reversibility/protocol/chronology before any promotion.
