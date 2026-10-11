# Cultivation PPO pilot v1

This directory preserves both the rejected eight-decision execution and the
scope-corrected campaign. The native simulator remains frozen at
`5559961164d6f9da8e6d20c1dc563ecec3eb04b7`; no native economics, ecology,
reward, observation, action, or physics code was changed.

## Rejected eight-decision execution

`failed-eight-step/` records the first complete three-seed run from learner
commit `a774a0c1a07487764c50082f4cde53af73b5281e`. Its aggregate evidence identity
is `e57c9d99a249291db54ad35d01b5ae88e03512f4a68412f858d4dd8e8b9b943c`.
Although replay, likelihood, numerical, checkpoint, KL, and reward-denominator
checks passed, the run is not promotion evidence: natural hunting experience
unlocked native husbandry during the longer rollouts, so excluded herd heads
became active and acquired nonzero gradients. The matching checkpoint bytes are
preserved outside Git in `checkpoints/cultivation-ppo-pilot-v1-failed-eight-step/`;
the paths embedded in the archived reports are their paths at execution time.

The rejected run also answers the trajectory question for that execution:
cultivation intensity did not steadily increase. Across seeds, mean requested
field operation moved from 0.4908 to 0.4763 and mean requested Development
budget moved from 0.4906 to 0.3892.

## Structural-horizon boundary

`structural-horizon-audit.json` simulates every one of the 300 frozen snapshots
for eight decisions under one fixed legal cultivation directive. All 300 are
clean through decision 3. One F15 case exposes husbandry capability at decision
4; herd ownership and herd action masks first appear at decision 6. Fishing
capability, fishing grounds, and movement remain absent throughout the audit.

The corrected campaign therefore uses three-decision episodes and evaluations,
with a fail-fast assertion on every reset and every post-step observation. The
assertion covers husbandry capability, owned herds, both herd masks, fishing
capability, fishing-ground presence, and movement. It supplements rather than
relies on the fixed-action audit. Training increases from 32 to 96 updates, so
each seed receives 2,304 in-scope transitions instead of 2,048. H=400 and the
declared optimizer remain unchanged.

The scope boundary is an experiment-design correction, not an architecture or
simulator modification. Longer-horizon future curricula must explicitly admit
husbandry or construct a new, physically valid native scenario cohort where it
cannot emerge.
