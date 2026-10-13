# Autonomous Elder collapse canary v2

User steering: economic failure (population × QoL × physical time) is acceptable
when caused by policy choices, not numerical saturation or inability to adapt.
This version replaces the earlier welfare/survival-only stop thresholds. Outcomes
remain prominent evaluation metrics, never rescue signals or privileged inputs.

## Bound and sampling
CPU only; immutable pilot finals17/29/43; at most32,768 additional focal samples
per seed,98,304 total, four native processes, one Torch thread per Python worker,
four hours maximum. Evaluate each seed every4,096 new samples. Stop any illegal,
nonfinite or replay error immediately. Native IPC has a deadline bounded by the
remaining campaign budget. No GPU work, merges, or baseline/artifact replacement.
All four curriculum families remain equally represented and all native movement,
trade, fission and migration processes remain live. Use the frozen native v4 bank
and unchanged reward/GAE/PPO settings, so this experiment isolates training health.

Freeze policy for each rollout. Each episode owns a generator keyed by training
seed and absolute episode index; within an episode only its own draws advance
that generator. Sort returned episodes canonically before PPO. Separate native
processes share no worlds. Validate one vs four workers and nonempty-Adam resume
before the canary, comparing actions, likelihoods, fingerprints and optimizer
updates. Changing completion order cannot change training. Checkpoint only after
complete episodes and updates, retaining model, optimizer, shuffle state,
episode cursor, counters, bank/config/source and parent-checkpoint provenance.

## Diagnostics and collapse criteria
Use a fixed train-only observation probe from twenty-four native episodes (world
seeds831001..831003; all four families), sampled by frozen pilot17 with sampler
seeds731 and732 per episode. No development data fits features or probe targets. Freeze
its serialized digest before canary launch. Track each actor and critic layer:
|tanh|>=.99 fraction, mean derivative1-h², constant saturated units, centered
activation effective rank and observation-dependent spread. Track active simplex
pre-clamp saturation, concentrations and analytic variance. Track legal categorical
entropy normalized by log(choice count), maximum probability and observation
sensitivity; exclude absent/singleton faces. Rare/disabled trade branches are
reported with eligibility counts, not mislabeled as dead parameters.

For actual PPO updates record actor/critic weight displacement, first-layer
gradients, head gradients, advantage/return variance, KL, clipping and fraction
of actor minibatches stopped by the KL guard. A stable action, concentrated
policy, or poor outcome is not itself a failure. Alerts (layer saturation>90%,
mean derivative<.01, active clamp fraction>10%, normalized entropy<.05, or
successive zero parameter updates despite nonzero advantages) trigger attribution.
Every scheduled evaluation includes the following recovery probes regardless.

Disposable clone probes: common nondegenerate workforce face (at least16 probe
rows, >=2 legal professions), balanced observation-dependent labels from the
most variable feature on that face; target requested shares alternate85% between
the first two legal professions, remaining15% shared over the other coordinates.
Use the actual masked Dirichlet likelihood through softplus and clamp. Fresh Adam,
lr.001,16 full-bank steps. Require finite gradients, relative likelihood-loss
improvement>=.5% (denominator max(1,|initial loss|)), and conditional mean-share
separation improvement>=1e-4. This separation measure prevents an output bias
alone from passing. Critic clone targets are its frozen original predictions
plus balanced ±1 labels, using the actual value path; 16-step Adam schedule with critic probe lr.0003.
Require MSE<.99, beating the best constant residual/intercept predictor (MSE1).
Never copy clone weights or optimizer state back into training.

A failed clone probe is repeated exactly on an untouched copy; a reproducible
failure pauses that seed for diagnosis. It is an operational no-go for continued
canary collection, not an automatic claim of irreversible collapse. Saturation,
clamps, feature/rank data and real update evidence distinguish numerical failure
from overly short probe budgets or legitimate convergence. Investigate and version
any remedy before continuing; do not relax thresholds silently. Include deliberately
saturated/dead and zero-learning-rate controls to prove detector sensitivity.

Record health at pilot entry, every4,096 samples, final boundary, and any stop.
Do not select a checkpoint using development performance. Failed seeds retain
their full evidence and do not prevent collecting independent healthy seeds within
the total cap. A watchdog and graceful checkpoint stop enforce the four-hour cap.
Inspect2× memory/latency changes against warmed fixed-workload preflight measurements;
stop unexplained resource growth. Increasing active world state must be documented,
not interpreted automatically as learner collapse.

## Attribution and promotion
Report welfare/food/population/event outcomes separately from numerical health.
Where economic results worsen but health passes, examine requested shares,
realized staffing/output and resource limits rather than repairing choices.
Movement/trade specialization may emerge; require repeated unseen-world learned
trade before claiming trade competence. This gate does not establish dynasty
control, broader-era coverage, optimal decisions or longer-horizon stability.
An independent reviewer assesses the collector, collapse evidence, any repairs
and final readiness. No larger campaign is implied by passing this canary.

Categorical recovery is also tested per prospective active head through its real
masked categorical likelihood, using a common legal face with at least16 rows
and balanced observation-dependent target choices. Apply the same16 steps,
.5% loss reduction and1e-4 conditional-probability separation improvement.
Counterparty/other heads lacking16 rows on a multi-choice legal face are reported
uncovered, never called collapsed. Their real eligibility/exposure remains in
health reports. Gate every covered head as well as workforce and critic.

Preflight expanded the original12-episode diagnostic probe to24 episodes before
launch to cover the offered-good head: the first bank had fewer than16 rows on
any common offered-good face. Both probes are preserved; only health-probe-v2
is the frozen canary gate. No canary outcomes informed this coverage change.

The prospective requested-good probe fixes the offered good to the first legal
stocked good from the observation, independent of model weights. Thus all
conditional probe faces, features, labels and targets stay fixed across checkpoints.

## Version2 preflight correction
The expanded192-row probe exposed fresh-Adam overshoot in pilot17's critic:
at lr.001 its MSE rose from1 to10.1664 after one step, then fell to1.1677 at16,
.8602 at32 and.3298 at64. With lr.0003 it reached.5104 at16 steps.
This is demonstrable recoverability, not a dead value network. Version2 freezes
critic probe lr.0003 before canary launch; actor probe and training learning rates
are unchanged. Preserve the failed v1 result and all attribution trajectories.

Probe each covered multipurpose activity face and each active two-coordinate
Beta control through its actual masked Dirichlet likelihood. Apply the same
actor loss/separation thresholds. Unavailable or singleton heads are reported
uncovered. Deliberately clamped recipe/control heads must fail this gate even
when workforce and categorical probes pass.

Also pause on eight consecutive actual PPO updates with nonzero critic residual
MSE (>1e-6), first-layer gradient (>1e-10), and zero critic displacement (<1e-12).
The corresponding actor condition requires nonzero advantage spread. A fitted
critic with negligible residual is not stalled. This checks retained Adam state,
which the disposable fresh-Adam probe cannot validate.
