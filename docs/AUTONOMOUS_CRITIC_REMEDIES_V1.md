# Discarded critic remedy comparison v1

The original six-checkpoint retained-Adam gate remains unchanged, including all
failures. After it finishes, pin all six raw diagnostics and checkpoints, probe,
runner and imported implementation hashes. Deduplicate critic targets only
within a checkpoint with identical ordered observation indices and labels;
assert their original 64-update MSE traces also match. Report multiplicities.

For every unique target compare exactly four variants: original LR .0003,
primary smaller-step candidate LR .0001, secondary comparator LR .00003, and
critic LayerNorm with LR .0001. Each starts from the same original checkpoint
weights, named Adam moments, shuffle state, clipping and 64 × 4 minibatch budget.
LayerNorm placement, identity affine initialization and epsilon match the
read-only audit v1. Only its new critic affine parameters lack prior Adam state.
Every variant uses the ORIGINAL unconverted critic value plus fixed labels as
its target. No candidate-centered target or DEV/HELDOUT fitting is permitted.

First require exact reproduction of the original 64-point MSE trace at LR .0003
for each target. Record entry error and the complete error trace for every
candidate. Prospective candidate criteria are final MSE < .9 AND maximum MSE
over the last eight updates < .9 on every target. The latter is an added
stability check for this new experiment, not a rewrite of the prior gate.
Do not substitute .00003 after a .0001 failure and label .0001 a pass.

Bound work to six checkpoints, at most 78 unique targets, four variants, 64
updates × four 32-row critic minibatches each (79,872 critic minibatches maximum),
one Torch thread and 300 seconds total. Zero native steps, actor updates or
action draws. Preserve all original files and failed/partial output; never save
or promote a replica. A success identifies a mechanically validated candidate,
not long-term native stability or scale-out readiness. Further native training
requires a separate explicit budget. First compare optimizer/architecture with
the eight-turn bank and reward horizon fixed; only then test 25-turn episodes
and a separately justified discount-horizon change.
