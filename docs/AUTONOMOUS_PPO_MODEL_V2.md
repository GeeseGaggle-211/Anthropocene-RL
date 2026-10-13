# Autonomous PPO model v2: categorical responsiveness remedy

The v1 CPU canary revealed a practical adaptation bottleneck: from seed17,
retained-PPO workforce reversal succeeded, while movement, quote enablement and
lot/price heads scarcely changed within64updates. This was not a globally dead
actor. The frozen failed tests and checkpoints remain intact.

Native action3/observation4/environment4/replay4 and the761features are unchanged.
Model v2 preserves the independent256x256tanh MLPs, workforce/recipe Dirichlet
and capital/livestock Beta heads. The old v1 model/trainer and five-head contract
remain separate and usable. New checkpoints identify autonomous-ppo-checkpoint-v2.

Every legal categorical distribution is p=.95*softmax(raw_logits)+.05*Uniform(legal).
Illegal choices remain exactly zero, absent/singleton branches have exact zero
log probability/entropy and consume no RNG. Sample, evaluate PPO ratios, and infer
using the identical mixed distribution. This supplies alternatives but is not by
itself a remedy for exponentially weak raw-softmax gradients.

Add .01*sum KL(Uniform(legal)||softmax(raw_logits)) to the actor loss. Compute it
with stable masked log-softmax. Its raw-logit gradient is q-Uniform, so it remains
useful when raw logits have become very uneven; no hard logit clamps are added.
Regularize structurally eligible faces, including prospective quote branches.
Average the requested-good regularizer uniformly over legal offered goods, so a
policy cannot avoid regularization by choosing a convenient upstream category.

Use exact joint categorical entropy:
H(move)+H(enable)+p(enable=1)*[H(partner)+H(give)+sum_g p(g)H(want|g)+H(lot)+H(price)].
Enumerate legal offered goods; retain upstream derivatives. Add continuous-head
entropy as before, coefficient.001. Exhaustive small-state enumeration checks
both entropy and gradients. This replaces v1's omitted downstream entropy.

Keep actor Adam lr.0003, reduce critic lr from.001 to.0003 to test value-update
overshoot, retain existing moments, gradient clipping, epochs, minibatches, GAE,
reward scaling and KL stop. Stop actor Adam momentum together with its KL guard.
Explicit SHA-checked warm starts retain episode/transitions/update counters and
sampler/shuffle state; no implicit architecture conversion or overwrite is allowed.

Validate on disposable retained-Adam reversal tests before any native continuation.
No artificial probe reward enters native training. A failed adaptation test needs
attribution, not threshold changes or claims that economic quality caused it.
Any native continuation must remain within the original aggregate98,304extra
transitions and per-seed32,768extra limits, including the already consumed40,960.
Thus seed17 has no native continuation allowance; seeds29/43 each have28,672left.
No GPU or broad training is authorized by this implementation contract.
