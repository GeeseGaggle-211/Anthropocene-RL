# Native-target critic calibration addendum v8.1

The initial v8 calibration remains archived as NO-GO: both H100 critics passed;
both H400 critics reduced error but failed the unchanged absolute normalized
bias threshold. Actors, actor Adam, RNG/counters and anchors remained protected.
The live critics showed approximately1% and0.35% tanh saturation, rather than a
hard numerical limit. No PPO started from that failed calibration.

Authorize exactly one closed-form TRAIN-only intercept fit per failed H400
critic. Delta=mean(fitting target minus fitting prediction), added only to the
live critic output bias. This is the exact unweighted squared-error minimum over
an additive intercept. No validation-based adjustment, clipping, sign choice,
search, additional epochs or rebuilt targets. Keep every other model tensor,
full optimizer, actor, anchors, RNG and counters unchanged. The critic-bias Adam
state remains unchanged and is explicitly recorded; this is not an Adam step.

Use the same original constant, entry-error and bias readiness criteria. Record
TRAIN residual/MSE before and after, requested and rounded applied delta,
protected hashes and native-target/checkpoint/config/source digests. Preserve
the original failed results. Zero extra native resolves and PPO transitions;
two closed-form fits consume the same cumulative1800-second CPU deadline.
Astra/high must review concrete source/config and then actual results before
matrix launch. Reaching readiness permits only the predeclared bounded matrix,
not scale-out or a GRU promotion.
