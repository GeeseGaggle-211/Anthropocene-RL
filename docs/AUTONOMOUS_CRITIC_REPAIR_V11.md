# Fixed current-policy critic repair V11

Fresh two-world endpoint point estimates pass: alpha pooled P2.4167 versus1.6667,
living endpoint Q.24657 versus.18825, mean original integrated welfare5023.235
versus4146.630, extinctions3 versus3. The declared gate still fails current-value
accuracy for101/211. They fail zero/double-tail variants too. A single P0=340
trade-family start886010 accounts for97.39%/94.61% of their squared errors;
101 settles no trade there. This is a context/forecast failure, not established
trade-credit failure, low-P0 aliasing, global saturation or a GRU requirement.

Preserve the passing307 checkpoint and all failed forecasts. Continue the V11
ledger from74949 resolves /2458.228495364019 active seconds, never reset caps.
One repair only: failed101/211 each gets nine25-decision TRAIN fitting trajectories,
maximum450 native+450 replay. Use the first six family starts in886001 plus
trade-d0 in886002/003/004, selected before collection for physical/P0 coverage.
New sampler offset610000. No fresh-world9/10 action collection or fitting; no
actor/PPO update, BC, optimizer reset, objective/input/output change or buffer refit.

Construct current-actor objective-specific fitting targets with frozen entry
critic tails and preserve rows/context/actions/likelihoods/values/receipts. Use
the existing fresh-world validation caches and their exact frozen targets/tails.
One TRAIN-only factor-aware intercept before and after exactly64 fitting epochs,
minibatch32, critic LR1e-4, inherited Adam, gradient clip.5, fixed shuffle offset
611000. There is no validation check/early stopping within the fixed fit and no
best-epoch selection. Protect actor/heads/actorAdam/RNG/counters/anchors/value
buffers byte-exactly. Save validation predictions for independent cached review.

After the fixed dose, reapply unchanged absolute metrics and nonworse-entry MSE;
report P0/family/seed and zero/double-tail diagnostics and TRAIN losses. These are
reused-validation, frozen-target repair metrics, not newly independent validation
or refreshed post-repair tail evidence. Endpoint outcomes remain attributable to
unchanged actors. The old307 checkpoint remains unchanged. Stage cap120 active
seconds within the existing V11 ceiling, no automatic until-pass retries. Astra/high
reviews concrete pins and actual repaired parameters/results before deciding
whether a bounded next training expansion is justified or more evidence is needed.
