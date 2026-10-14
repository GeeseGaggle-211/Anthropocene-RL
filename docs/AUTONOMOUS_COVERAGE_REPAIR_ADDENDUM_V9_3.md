# Frozen-actor critic coverage repair v9.3

V9.2 mechanically passed but all three strict critics failed world-disjoint
validation despite near-interpolating only96/94/98 fitting rows. Preserve failed
checkpoints, targets, configs, source pins and ledgers. No v9.2 PPO or DEV took
place. Final cumulative ledger19003resolves/667.0512370969955activeelapsedseconds.
This prospective repair expands genuine native fitting data without weakening
EV>=.60/MSEvariance<=.40/absrawbias<=2/nonworse-original gates or adding recurrence.

Use every available world-family slot across fitting worlds881001..004:23slots,
all six families, preserving the absent world4 fishing trajectory. Predeclare
reset ages0/2/8 and require actual native admission. For each original fresh
warmed actor101/211/307, use sampler seeds episode_seed(seed+166000,slot_index),
distinct from the original seed+60000 collection. Maximum23*25*3=1725additional
native+1725replay. Join original fitting rows with new fitting rows. Retain exact
six-family original world5/6 validation trajectories and targets; label reuse
as calibration-gate validation, not a new untouched test. No DEV/HELDOUT fitting.

Reset only live critic tensors from its frozen initial_normalized_critic, clear
only live critic Adam, and set value_offset0/value_scale1. Because residual
subtraction precedes addition, this recovers original reference values exactly.
Prove actor/heads/anchors/actor Adam/RNG/counters/config unchanged, including
populated-Adam test. Collect new targets under exactly that original prefit
value function, preserving signed frozen tails and explicit zero/double-tail
and H100 zero-tail sensitivity. Native raw rewards are nonnegative, while a
signed frozen tail can make low-value censored targets negative; do not clip them.

Fit new fixed global TRAIN-only mean/max(std,1) on the union fitting targets,
then freeze buffers. Same model7 floor.1/native6/features1062, critic LR.0001,
maximum256epochs, TRAIN-only before/after intercept, strict early-stop/final
gates, protected actors/Adam/RNG/anchors/normalizers. No actor re-pretraining or
compiler changes while reusing validation. All three must pass before PPO.

Reallocate unused v9.2 future resolves prospectively; the original ledger/caps
remain immutable. Retain six equal LR-paired cells but contract from1024 to448
transitions each,224-transition rollouts (episode25,H400), total2688retained.
Three baselines and six finals still use the same six DEV starts,25decisions,
maximum1350+1350replay. This smaller canary checks numerical trainability and
retained-optimizer reversibility, not economic competence or broad scale-out.
Retain matched initialization/reset/sampling state except actor LR.0003/.00003.

New maximum1725+1725+2688+1350+1350=8838resolves→27841cumulative,150reserve
under unchanged27991ceiling. New active elapsed deadline is2369.4072460189927
minus chained prior time;4nativeworkers/1Torchthread, per-neural-minibatch and
native-request deadline checks. Store any failure, never replenish stage caps.
Read-only admission adds no resolves and chains actual elapsed time before the
training repair. Astra/high must clear exact pins before admission/repair and
independently judge actual numerical/economic results before promotion.
