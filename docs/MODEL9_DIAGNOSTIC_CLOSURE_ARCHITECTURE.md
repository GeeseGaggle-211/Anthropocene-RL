# Model9 diagnostic closure architecture

Status: Astra High approved for diagnostic-only implementation, 22 October 2026.
This approval does not authorize a VM start, training launch, checkpoint
promotion or model-architecture change.

This document closes the four evidence gaps identified after T3. It supplements
`MODEL9_ECONOMIC_OBSERVABILITY_ARCHITECTURE.md`; that document remains normative
for the economic sidecar and prospective training trace. The gates run in order.
A later gate may not reinterpret an earlier incomplete result as a pass.

## 1. Authority, evidence classes and sampling

All diagnostics are opt-in and reporting-only. They do not change policy inputs,
actions, rewards, simulator state, RNG streams, training batches, optimizer order,
checkpoint selection or promotion.

Every row and report declares one evidence class:

1. `historical-record`: immutable contemporaneous artifacts from the original run;
2. `checkpoint-evaluation`: a bound checkpoint evaluated now on a frozen cohort;
3. `instrumented-replay`: a historical action/background sequence replayed from a
   verified snapshot and producer;
4. `matched-intervention`: independently named continuations from one verified
   snapshot under a registered action intervention; or
5. `prospective-update-audit`: a new, explicitly nonqualifying diagnostic update.

Reports never merge evidence classes into one estimator. Environment replay
supports reconstructed simulator/economic evidence only. Historical PPO causality
requires contemporaneous training evidence. An exact optimizer reconstruction
would need a separate reviewed protocol binding the pre-update actor, critics,
normalizer, optimizer moments/steps, every RNG state, ordered minibatches,
auxiliary inputs, runtime and verified post-update hashes; recomputed quantities
would still be labelled reconstructed. A current checkpoint evaluation is never
relabeled as the optimizer update that happened during T3.

Local training workloads are stability/validation probes only and use at most
three optimizer seeds. They cannot choose an arm, tune a treatment, establish
readiness or promote a checkpoint. Readiness-bearing training comparisons run on
the VM as at least three matched arms by five paired optimizer seeds. Historical
T3 documents currently name disjoint optimizer seeds 70-74, 75-79 and 80-84,
while the later reviewed protocol names paired seeds 170-174. These are not
interchangeable: original manifests must be reconciled; historical results use
unpaired optimizer uncertainty if that is what the artifacts prove; and a future
matched cohort receives its own resolved authority. A different design requires
reviewed preregistration. Confirmation seeds 3000-3059 are denied everywhere.

The VM may not be started and no job may be submitted without explicit user
approval. Every report declares its execution tier and whether selection/readiness
claims are permitted.

## 2. Gate protocol and review packet

Each gate writes a content-addressed review packet containing:

- architecture and implementation commit IDs plus dirty-state proofs;
- complete CLI, expanded seed/snapshot lists and canonical configuration;
- simulator, bridge, learner, checkpoint, normalizer and input hashes;
- raw-row hashes, deterministic aggregate hashes and completion marker;
- tests, coverage, missingness, performance and memory results;
- supported, unsupported and unresolved claims; and
- a one-page `PASS` or `HOLD` decision report.

After implementation and collection for each gate, Astra High independently checks
the packet and raw evidence and reports model-architecture and training shortfalls.
Only Astra `PASS` permits the next gate. `HOLD` freezes interpretation and triggers
a reviewed design correction; it never authorizes adaptive tuning on the cohort.

## 3. Gate 1: initialization-loss decomposition

### 3.1 Question and lineage model

Locate the first detected behavioral and outcome change across the actual artifact
history. The user-facing comparison is interested in the prior policy, any actual
conversion, the guarded clone and the trained T3 policy, but those names do not
establish a linear parameter lineage. A prior policy can be an evaluation reference
or teacher-data source without being the clone's parameter parent. A historically
nonexistent conversion is `not-applicable`; it is never manufactured to complete a
four-column report.

The comparison distinguishes format/schema migration, actor conversion, normalizer
change, imitation fitting, critic initialization and PPO continuation. It never
infers an edge from filenames or today's loader behavior.

### 3.2 Frozen artifact DAG

`initialization_lineage.json` is an artifact-backed directed acyclic graph. Each
node binds stable ID/role, checkpoint path and SHA-256, raw schema, actor/critic/
normalizer contract, producing learner identity, original producer/run evidence
and every source-artifact hash.

Each edge has one registered relation: `parameter_parent`, `teacher_data_source`,
`evaluation_reference`, `schema_conversion`, `normalizer_change`, `clone_fit`,
`critic_initialization` or `ppo_continuation`. An edge may also carry an equivalence
claim only after declaring:

- source/destination schemas and numeric runtime;
- mapped legacy tensors/outputs and appended outputs;
- action decoder, active/delegated heads and normalizer mapping;
- exact tensor preservation, mapped-distribution numerical equivalence,
  deterministic behavioral equivalence or stochastic behavioral equivalence; and
- tolerances and corpus frozen before results are inspected.

Exact tensor copies are checked separately from forward computations. Widened
matrix execution is not assumed bitwise because new columns are zero. New heads
must remain disabled or delegated for an old-policy behavioral-equivalence claim.
Corpus equality is evidence on that corpus, not proof over all observations.

The strict diagnostic loader inspects raw checkpoint metadata before model
construction, rejects unknown/future schemas and requires a hashed migration and
default-resolution record. Source-side inference uses a pinned historical
implementation or reviewed adapter; both sides may not silently pass through the
current `upgrade()`. Environment overrides and region/head defaults are explicit.
Cycles, missing required nodes, mutable inputs and duplicate hashes under conflicting
semantics are rejected. A reconstructed conversion is a separate reconstructed
node and does not prove that conversion occurred during T3. Missing required
artifacts remain `missing`; summary JSON cannot close them.

### 3.3 Evaluation panes and producer compatibility

The primary pane preserves each node's historical head activation and decoder. A
second `standardized-flags-intervention` may place compatible actors under common
flags, but it is not substituted for historical semantics. Both panes freeze world
seeds, reset snapshots, warm-ups, horizon, background policies, personality,
action mode and any stochastic-policy RNG schedule. Training, selection,
calibration and confirmation families are excluded.

Historical producer provenance is bound separately from the live evaluation
producer. A common producer is allowed only through reviewed semantic-comparator
evidence covering reset/configuration, action, observation, reward and state
trajectories. Otherwise the node is incompatible and the gate is held. Worker
contracts are never rewritten or bypassed. A legacy external evidence wrapper may
bind contemporaneous records but cannot claim modern qualification.

The evaluator adds capabilities absent from current deterministic `rollout`:
reset-snapshot input, explicit deterministic/seeded-stochastic mode, canonical
trajectory digest and integrated population/QoL-year metrics. Registered
equivalence edges compare only their declared mapped outputs, masks and decoded
actions. Tensor preservation is bitwise; mapped forward/behavior comparison uses
the frozen numeric rule. A failed expected-equivalence edge stops descendant
interpretation.

For every comparable node and world, report welfare, population-years, ending
residents, QoL-years, trade utility, extinctions, settlements, service outcomes,
compiler interventions and trajectory digest. Aggregate paired world differences,
ratios and intervals without independently pooling optimizer/world identities.
Ordered outcome deltas telescope descriptively; they do not causally isolate the
architecture, normalization, data and optimization changes bundled in an edge.

### 3.4 Gate 1 acceptance

Gate 1 passes only when the inventory proves every actual required node/edge,
registered equivalence edges pass, and each comparable node completes its
applicable historical and/or standardized cohort. Behavior and outcome are each
classified as `first-detected-change`, `no-detected-change` or
`insufficient/incompatible`; no changed boundary need be invented. Exact conversion
failure or missing mandatory lineage evidence is `HOLD`. A learned guarded clone
is expected to differ and its delta is measured, not treated as conversion failure.

## 4. Gate 2: causal economic accounting

Gate 2 implements the terminal links specified by the economic observability
architecture and produces a decision-grade controllability matrix. Every active
control row joins in native units:

`actor distribution/sample -> native demand -> actor-capped request -> compiler
grant -> executed worker/input claim -> committed physical receipt -> inventory
change -> consumption/final service -> population/QoL outcome`.

Manufacturing preserves good/route/recipe identity and separately attributes
opening-stock draw, intermediate consumption, export reservation, storage block,
surplus and final-service fulfilment. No mixed service scalar is invented and no
ratio crosses units. Trade spans requisition, eligibility/gating, partner search,
offer, counter/ignore/rejection reason, handshake, stock versus made-to-order
supply, escrow/claim, production, delivery, storage acceptance, refund/return and
immediate/delayed utility. Rejected potential is not inventory flow.

Each matched intervention preregisters one estimand: clamp one emitted simulator
control; alter one action distribution; or alter one conditional policy mechanism.
It freezes treatment timing/duration, other emitted controls, subsequent policy
behavior, commitment handling, joint support and whether background actors replay
actions or react under a frozen policy. Region is sampled before and conditions
economy mix, while commitments can activate/suppress capital controls; an upstream
intervention is not a one-head effect when downstream controls change. Common RNG
coupling is claimed only for streams the simulator can keep aligned and includes
draw/state divergence diagnostics.

Request-to-receipt allocation for shared goods is deterministic bookkeeping, not
causal identification. Causal claims use matched effects. Coverage denominators
and thresholds, including inactive/unsupported domains, are frozen before results.

Gate 2 passes only when mandatory population, food, Stone/Clay/Wood, craft-good,
herd, field and bilateral-trade conservation terms are available and reconcile;
terminal linkage meets each preregistered domain threshold; matched-intervention
support/divergence is reported; and unsupported rows have explicit reasons. Zero
residuals in an incomplete subset cannot pass.

## 5. Gate 3: independent critic measurement

The scorecard freezes states before results and binds each objective's reward and
shaping definition, state/population unit conversion, personality, continuation
action regime, lineage transition semantics and discount. It records the
state-value prediction and then runs no-learning continuations. Bootstrap-free
realized returns are the scoring targets.

The critic is a state-value model and cannot directly rank actions at one state.
An action-ranking prediction is explicitly one-step
`reward + gamma * V(successor)`, with registered descendant aggregation, and is
scored against independent matched continuation returns. The prediction may use
the critic bootstrap; the target may not bootstrap from the critic under test.

Every prediction-target accuracy or calibration metric uses the critic's declared
return horizon and units. Terminal returns support point metrics. Independent tail
estimates support separately labelled estimated-target metrics with propagated
uncertainty. Tail bounds support interval-valued errors or bounds on metrics; their
midpoint is never silently treated as the target. Unresolved tails withhold
full-value bias, MAE, RMSE, explained variance, calibration slope/intercept, sign
and ranking wherever the requested conclusion is not identified. Prefix-only
quantities remain separately labelled and do not establish full-value accuracy.
Stochastic continuations use frozen replicate counts and uncertainty rules.

By objective, action regime and constraint, report counts, means, bias, MAE, RMSE,
explained variance, calibration slope/intercept, alternative-ranking accuracy,
rank correlation, matched-effect sign agreement, tail coverage and all exclusions.
Lambda-target fit remains a separately labelled training-consistency table.

Gate 3 passes when independent targets have complete provenance, frozen minimum
coverage/tie/ambiguous-effect rules pass and enough matched alternatives measure
ranking/sign agreement. Accuracy is not required: a localized measured critic
failure is a valid gate result. Missing or self-target-contaminated evidence is not.

## 6. Gate 4: optimizer-direction audit

Gate 4 uses fixed audit states and disposable checkpoint copies. Actor, value
critic and cost critic own separate encoders/optimizers. The report verifies actor/
critic parameter overlap as zero and never describes value-critic gradient conflict
as direct actor-parameter interference.

Before and after each audited update, report per active actor head:

- distributions, masks, sampled/deterministic actions, entropy and log probability;
- log-density or preregistered neighborhood-probability change for independently
  beneficial/harmful alternatives, with latent/action transforms specified;
- raw, scaled, clipped and weighted objective, guard, regime and teacher terms;
- PPO ratio/clipping/KL and actor gradient/update norms by head/shared trunk;
- per-loss actor gradients on the actor trunk and per-objective value gradients on
  the critic shared encoder, with norms/cosines; and
- auxiliary/search/teacher losses and verified parameter sharing.

Every row binds Adam moments/step counters, optimizer hyperparameters, RNG states,
ordered rollout/branch/search minibatches and before/after hashes. A weights-only
or fresh-optimizer audit is separately named prospective evidence. Isolated
component-gradient capture restores `.grad`, RNG and optimizer state. Component
gradients describe geometry, not additive attribution of the clipped combined
gradient or nonlinear Adam update, which are reported separately.

Region-conditioned mix and commitment masks remain explicit. Beneficial/harmful
labels use frozen objective/personality/guard weights and Gate 2-3 uncertainty.
Joint PPO advantage is not presumed erroneous; the audit tests whether the realized
joint update moves controllable distributions in independently supported directions.

Gate 4 passes when rows bind full update state, the preregistered supported-head
direction-coverage threshold passes, unsupported heads are explicit, value-head
gradient geometry is measured at the critic encoder and deterministic re-analysis
reproduces the report. A measured misdirected update is a successful diagnosis;
missing required evidence is `HOLD`, not a blanket `insufficient` pass.

## 7. Performance and validation

Diagnostic-off simulator and policy trajectories remain bitwise identical. Capture
is bounded by active villages, stable goods, receipts, transactions, audit states
and selected parameters; it does not scan world tiles per village or retain
unbounded graphs. Existing observability throughput/RSS budgets remain mandatory.
Before collection freeze maximum audit-state count, supported heads/loss terms,
continuation horizons/replicates and worst-case retained-autograd memory. Streaming
ledger rows alone does not bound continuation or gradient cost.

Tests cover schema round trips, missingness, stable IDs, fork isolation, replay,
unit-safe funnels, complete trade failure paths, conservation, lineage rejection,
strict no-upgrade loading, mapped conversion equivalence, independent returns,
tail withholding, per-head density deltas, non-mutating gradient decomposition and
deterministic aggregation. Cross-repository tests bind the exact simulator DLL and
both source commits.

## 8. Deferred decisions and carried shortfalls

No gate automatically deletes critic heads, changes actor objective weights, adds
personality attention, changes FiLM, tunes entropy/KL/clipping/LRs, or alters
compiler admission. After Gate 4, Astra reviews the combined evidence. Any Round 4
treatment receives its own reviewed preregistration and full paired VM cohort.

Every review carries these known shortfalls until evidence resolves them: token
cloning changes representation and normalization; hierarchical decoding can change
behavior without changing learned weights; scalar-to-vector conversion creates new
non-welfare heads; joint PPO credit does not identify each head's marginal economic
effect; value-head gradients can conflict inside the critic encoder; and missing
historical optimizer state cannot be repaired by present-day evaluation.
