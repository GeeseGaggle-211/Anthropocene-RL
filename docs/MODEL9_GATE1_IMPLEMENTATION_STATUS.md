# Model9 Gate 1 implementation status

Status: the Gate 1 fail-closed validation framework and the Astra-authorized
diagnostic evaluator capability phase are implemented for independent Astra
High re-review, but implementation-quality approval has not been issued and
production evidence remains incomplete. Evidence
collection and post-collection review have not run. The gate remains `HOLD`. This work
does not authorize a VM start, training launch, checkpoint promotion,
model-architecture change or implementation of Gates 2-4.

Authority: Astra High approved the diagnostic-only architecture on 22 October
2026. The implementation binds the approved architecture revision
`a3f64427d536e66eab9e0a016d1e9c2ef5f9a8f0` and the evidence classes in
`MODEL9_ECONOMIC_OBSERVABILITY_ARCHITECTURE.md`. This is implementation authority
only; the architecture's explicit VM, training, promotion and model-change
prohibitions remain in force.

## Implemented surface

`anthropocene_rl.autonomous_v7.initialization_lineage` provides three commands:

```text
python -m anthropocene_rl.autonomous_v7.initialization_lineage validate --manifest initialization_lineage.json --out validation.json
python -m anthropocene_rl.autonomous_v7.initialization_lineage evaluate --manifest initialization_lineage.json --node NODE --pane PANE --out evaluation.json
python -m anthropocene_rl.autonomous_v7.initialization_lineage packet --manifest initialization_lineage.json --evaluations evaluations.json --verification verification.json --implementation implementation.json --out PACKET_ROOT
```

The manifest is schema 1, kind `initialization_lineage`, and is sealed by a
canonical `digest`. A separately content-addressed lineage inventory protocol is
bound to an independent Astra High review artifact and prescribes the exact
nodes, edges, user-facing slots, applicability panes, chronology, branches and
run identities. It requires the historical T3 inventory to contain all three
arms and their five disjoint, unpaired optimizer seeds: T3per 70-74, T3def
75-79 and T3lo 80-84. The shared prior-to-clone boundary and each
clone-to-trained branch remain separate paths. It explicitly registers required
node and edge IDs. A node
records its role, evidence class, status, content-addressed checkpoint, raw
schema, actor/critic/normalizer contracts, producing learner, producer/run
evidence, all source artifacts, historical status, loader implementation,
default-resolution record and complete historical actor/decoder semantics.
Missing and historically nonexistent artifacts are records with `missing` and
`not-applicable` status; they cannot disappear behind a summary. Only records
listed in `required_nodes` or `required_edges` block on those statuses. Optional
missing or not-applicable history remains explicit and nonblocking.

A second independently reviewed, content-addressed evaluation/claim protocol
freezes the complete pre-result node and edge records, checkpoint and source
artifact identities, producer/run/migration evidence, every equivalence
declaration, and the full pane/world cohort and semantics. Its reviewed proposal
digest therefore changes if a checkpoint, cohort seed or horizon, pane,
historical producer, implementation, corpus, tolerance or claim is substituted.
Post-execution result references are the only equivalence fields intentionally
excluded from this preregistration projection.

An edge uses exactly one of `parameter_parent`, `teacher_data_source`,
`evaluation_reference`, `schema_conversion`, `normalizer_change`, `clone_fit`,
`critic_initialization` or `ppo_continuation`. Equivalence claims use exactly one
of `exact_tensor_preservation`, `mapped_distribution_numerical_equivalence`,
`deterministic_behavioral_equivalence` or
`stochastic_behavioral_equivalence`. Each claim freezes schemas, numeric runtime,
tensor/output scope, appended-output disposition, decoder, active/delegated
heads, normalizer mapping, tolerances and corpus. Exact tensor claims are
recomputed bitwise from raw checkpoints. Forward and behavioral result artifacts
are not accepted as summary assertions. A claim binds a typed canonical corpus,
numeric runtime, source and destination inference implementations, and a named
executor implementation. Raw execution receipts must cover the exact Cartesian
product of corpus case, mapped-output scope and source/destination side. The
reviewed execution protocol registers an exact output contract for every
case/scope: float dtype, tensor shape, ordered component labels, mask shape and
labels, and a bounded recursive decoded-action schema. Float values must be
finite, in range and canonically representable after deterministic float32 or
float64 packing; comparisons use those normalized declared-dtype values rather
than unconstrained JSON doubles. Parsed JSON integers retain their exact
arbitrary-precision value through this check: the integer reconstructed from the
declared float must equal the original integer, so Python's intermediate float
conversion cannot hide precision loss at float32 or float64 boundaries. Each
execution and raw output repeats a
side-specific context digest over the claim/protocol, case, scope, endpoint
checkpoint, input, output contract, producer/executor implementations, runtime
and RNG schedule. A source output reference therefore cannot be substituted for
a destination output and resealed as evidence. The validator recomputes numeric
errors, masks and decoded-action equality. Seeded-stochastic claims additionally
bind an exact per-case RNG schedule. All nested schemas, executions and raw
outputs are preserved transitively in the packet CAS.

The validator:

- inspects checkpoint hashes and weights-only raw metadata before model
  construction;
- requires the declared raw schema to equal the complete inspected schema, not
  an empty or favorable subset;
- rejects absent, unknown and future contracts instead of applying the legacy
  contract-8 default;
- never calls `upgrade()` and requires a hash-bound default-resolution record
  with an empty `implicit_defaults` list;
- rejects cycles, omitted required records, mutated/hash-mismatched inputs,
  conflicting semantics for one checkpoint hash and confirmation seeds
  3000-3059;
- requires schema-conversion records to bind both endpoint hashes and contracts;
  reconstructed adapters must identify their implementation and explicitly deny
  historical conversion; and
- keeps each row in one evidence class.

The two evaluation panes are distinct. `historical-semantics` always uses the
node's bound flags and decoder. `standardized-flags-intervention` must name its
override and intervention label. Both freeze evaluation-only world families,
warm-up, horizon, background policy, personality, action mode, environment,
worker contract and requested metrics. Training, selection, calibration and
confirmation seed families are rejected.

Every pane and evaluation report declares `local` or `vm` execution and
affirmatively denies selection and readiness claims. Service outcomes use a
content-addressed pane schema with an exact field registry and native unit for
each service. Each observation is either a finite value or an explicit null with
a nonempty availability reason. Null remains valid missingness evidence but
blocks the gate; omitted/extra fields and unregistered raw dictionaries are
rejected. Count-valued outcomes require nonnegative integers.

The current evaluator supports one explicit background declaration only:
`kind=none`, because the checkpoint actor controls every live village. It
supports one global frozen personality vector and rejects personality maps
instead of replacing them with `None`. Evaluation reports repeat and bind both
declarations. Packet ingestion reconstructs the exact requested world contract,
including movement/head semantics, snapshot identity, turn, players and cohort
digest, and verifies the reported producer's simulator, bridge, flags and raw
material state against the frozen worker contract.

Learned movement and regional movement are independent semantics. The
`learned_movement` flag controls whether rollout movement is policy-driven;
`region_movement` controls whether that learned decision uses the regional
decoder instead of the legacy movement head. World descriptors bind the former,
while strict actor configuration binds the latter.

The live evaluator now executes the architecture's complete mandatory metric
surface. It passes content-verified snapshots through both worker validation and
`reset(snapshot=...)`; the reset sidecar is required to be empty and the first
recorded transition is the successor of the restored boundary. Deterministic
and seeded-stochastic modes are explicit. Stochastic execution uses a fresh CPU
`torch.Generator` per frozen world/node/step, verifies its seed-derived state hash
and the exact active-head draw order, recomputes the post-draw state from the
actual number of action-bearing rows, and leaves process-global Python, NumPy
and Torch RNG streams unchanged. A pane-wide schedule cannot stand in for nodes
with different active policy heads.

Both live and imported reports pass through one schema-2 raw-trajectory
validator. The retained reset record separately binds the reset trajectory ID,
reset fingerprint and a contiguous reset-to-opening warm-up chain; economic rows
retain that reset trajectory ID while policy continuity uses the evaluation-opening
fingerprint. The opening boundary retains the initial policy observation that
produced the first evaluated action. Steps have contiguous indices,
unique bound transition IDs, exact source/successor continuity, action request
and decoded action, stochastic receipt when applicable, raw economic rows and
an explicit termination/truncation reason. A short trajectory is accepted only
for an evidenced all-villages terminal boundary; zero steps, duplicate restored
transitions, a missing opening observation, a missing economic camp-turn, or an
unevidenced short horizon fails closed. Reviewed capture limits cover step,
village, economic-row and canonical-byte counts. Canonical-byte accounting
includes the reset record and every retained warm-up-chain link as well as the
opening and transition rows. Live capture checks each prospective warm-up link,
opening and transition before retaining it; imported and CAS replay recompute
the same complete receipt.

The reviewed world record now freezes the reset turn separately from warm-up;
snapshot worlds must bind the snapshot descriptor turn to that reset turn, and
live execution checks the actual restored/reset boundary before warm-up. The
retained opening turn must then equal reset turn plus warm-up. Each pane also
freezes the current bridge observation core width. Every live village must carry
the exact contract-12 observation keys, finite core/set values, exact row and
head masks, and a recomputed observation digest. Every decision-bearing source
camp must have exactly one complete action request/decoded action for its active
historical or standardized heads. Categorical routes, site/region movement,
economy mix and commitment holds are checked against the exact source masks;
null, masked-out or arbitrary action values are not evidence. Each action now
retains a digest-sealed decoder receipt containing the exact typed float32/int64
head outputs or stochastic latent draws, source masks, batch row/shape, active
mode/head semantics and emitted bridge action. Imported evidence replays the
same production `act_deterministic` or PPO `to_bridge_actions` helper forward
and requires exact canonical output equality. It never reconstructs the hidden
three-simplex from the two rounded bridge shares. Batch shape and row are bound
so native vectorized float32 behavior is reproduced without retaining or
trusting unrelated rows. The receipt also binds the exact positive,
non-overlapping bounded tensor strides: native actor head views are not silently
made contiguous, because that can change a float32 sigmoid result by one ULP.
Before any NumPy or Torch construction, the validator recursively checks the
original JSON scalar leaves and rectangular shape. Booleans cannot enter numeric
tensors, integer tensors accept only in-range JSON integers, and a JSON integer
in a floating tensor must be exactly representable in the declared float dtype;
heterogeneous, ragged, nonfinite, overflowing and underflowing values reject.
Deterministic `.01`/`.99` snapping, categorical masks,
the `.5` pure-fishing fallback and forced region selection are therefore checked
as forward decoder consequences. Seeded-stochastic receipts retain unsnapped
semantics and bind the exact frozen draw/mask surface. A missing, substituted or
resealed raw decoder input fails live/imported validation and strict CAS replay.

Transition coverage is derived from the validated source observation/action
surface, never from successor `priorPopulation`. Source camp, society and
population identities must agree with each economic row and successor witness.
Every decision-bearing camp remains present as either a live or explicit
zero-population terminated successor. New camps require a typed raw
founding/fission transfer witness that binds source/destination camps, parent
society and transferred population; an existing camp cannot be reflagged as
founded and a one-sided migration cannot invent a new destination. Native
migration is represented by identical endpoint copies. The validator derives a
canonical source/destination/people/phase identity, requires exactly one copy on
each existing endpoint, rejects mismatched or aliased endpoints, and reconciles
emigration/immigration with opening residents, pre-migration linked population,
ending residents/parties and final linked population. Those population
equations run for every camp even when the recognized migration total is zero;
deleting or renaming both endpoint copies cannot suppress nonzero raw migration
receipts, and unregistered transfer kinds reject rather than disappearing from
reconciliation. Reset/opening metadata and genuine daughter boundaries may report
zero years, but economic transition durations remain finite and strictly
positive. A live successor cannot substantiate
`all-villages-terminated`, and unexplained disappearance, identity substitution
or `priorPopulation` suppression rejects the report.

Economic diagnostics are opt-in only when a requested metric requires them;
ordinary rollouts neither enable the simulator sidecar nor deep-copy/retain the
Gate 1 trajectory surface.
Population-years sum the simulator's committed `personYears` values in declared
`person-years`; QoL-years sum committed settlement QoL times authoritative
transition duration in `qol-years`. This is Gate 1 evaluation accounting, not
the proposed Section 8 training objective. The canonical trajectory digest
binds ordered pre/post fingerprints, policy-facing state, decoded action,
outcome/reward/terminal fields and the complete validated economic sidecar.
Entity collections with semantically irrelevant ordering are canonicalized by
their registered identity, while step and action component order remain
sensitive.

The registered 41-field service surface preserves cultivation and herding food,
all eleven named manufacturing goods, Stone/Clay/Wood receipts, settled trade,
realised trade utility and imported/exported quantities for all eleven named
trade goods. Every field retains its native unit and identity; zero is observed
zero, while unavailable data carries a nonempty reason and holds production
acceptance. Compiler constraints retain raw transition/camp/source rows and are
summarized by named domain and reason without mixing service units.

The authoritative raw-economic validator is shared by live capture and imported
packet replay. It requires the exact envelope/domain schema, trajectory/turn/camp
identity coverage, stable goods/material identities, finite values, positive
durations, count integrality and physical nonnegativity while retaining signed
trade utility. Missing collections never become observed zeros. All mandatory
headline outcomes—welfare, ending residents, trade utility, extinctions,
settlements, population-years, QoL-years, services, compiler counts and the
trajectory digest—are recomputed from this raw evidence; conflicting report
summaries are rejected. Boundary trade utility must equal the corresponding raw
signed `domains.trade.realisedUtility`; aggregation uses that raw value rather
than a second trusted boundary total.
Every raw unit declaration is checked without mutating imported or CAS evidence
against a present economic field and the
native unit vocabulary; the exact units of all Gate-1-consumed scalar fields are
then required, including signed realised trade utility in
`person-qol-years`. The opt-in bridge only supplies these registered labels when
the older simulator envelope omits them and rejects conflicting or unknown
labels. A resealed `bananas` trade unit is therefore rejected rather than
silently accepted or relabelled.

Current-contract inference remains strict-loaded. Current and reviewed-legacy
loader eligibility runs inside success-and-exception-safe preservation of the
process-global Python, NumPy and Torch RNG streams; manifest validation,
imported report ingestion and packet construction therefore cannot advance
those streams merely by constructing or rejecting an actor.
Strict actor loading requires exact raw state-dict key coverage before PyTorch
load hooks run, so compatibility hooks cannot synthesize absent FiLM or region
tensors. Tensor-equivalence declarations resolve a top-level checkpoint mapping
plus ordinary dotted PyTorch state-dict keys and reject ambiguous paths. Strict
native normalizer loading accepts only the two actual state schemas: complete
unguarded moments, or complete guarded moments plus `guard` and an exact boolean
`*_dead` mask for core and every registered entity set (and the privileged block
when declared). Means and strictly positive standard deviations must be finite
canonical float32 lists with exact widths; guard must be a finite nonnegative
float. Partial guards, missing/extra masks, integer or wrong-shaped masks, extra
keys and silently coercible moment values reject.
The evaluator also contains a pinned contract-8 actor/input/decoder implementation
transcribed from historical commit `2f4cc895e6bcd897200d340851d69be98fd18e00`.
It selects only the historical core prefix and five contract-8 entity sets,
requires exact raw actor and normalizer keys/shapes, and never calls `upgrade()`
or constructs post-contract-8 tensors. Use is fail-closed behind an independently
reviewed, content-addressed adapter protocol binding the historical source,
executing source hash, input mapping, decoder defaults and checkpoint-specific
default-resolution/migration records. It cannot claim a historical conversion.

Every evaluation requires the exact checkpoint worker contract or complete reviewed producer
comparison over reset, configuration, action, observation, reward and state
trajectory semantics. The same historical-to-live check is recomputed for
imported reports, as is the checkpoint's strict/native or reviewed legacy-loader
eligibility, so a resealed report cannot bypass producer or loader compatibility. It
does not silently substitute a fresh rollout or final snapshot statistic when
asked for an unsupported capability.
Loader eligibility is also checked during manifest validation itself: exact raw
actor keys, shapes, dtypes, devices/layouts and normalizer keys/moments must be
loadable before any report can be accepted. Strict packet reconstruction resolves
the reviewed contract-8 adapter implementation and its source bytes only through
the packet CAS. It verifies the repository-relative source identity and digest
without consulting a still-readable live checkout; a missing or corrupted
preserved adapter source fails before evidence interpretation.

Review packets are stored under their canonical packet digest and contain CLI,
expanded worlds, artifact hashes, raw-row and aggregate hashes, equivalence
checks, verification evidence, missingness, supported/unsupported claims, separate behavior and
outcome classifications, a one-page decision, and a completion record binding
the output files. The implementation input is itself a content-addressed,
digest-sealed `gate1_implementation_identity` artifact. It records the clean Git
commit, exact source-file hashes and Git-porcelain dirty-state proof for the Gate
1 modules, evaluator, focused capability/lineage/bridge tests, this status file and both governing design
documents. Packet construction rehashes those executing sources and recomputes
the commit and dirty-state proof. An omitted source, dirty tree or identity/hash
mismatch is `HOLD`; caller-supplied assertions cannot establish implementation
identity. Test success is not accepted as a caller assertion. The manifest
instead binds a digest-sealed `gate1_verification_protocol` whose coverage,
performance and peak-memory thresholds are frozen independently of results. A
verification artifact binds the implementation commit and protocol, declares
its nonselecting execution tier, and includes hash-verified test output plus
registered raw coverage, performance/machine and peak-memory reports. Nonzero
test exits, sub-threshold coverage, required missingness or an exceeded budget
holds the packet. Packet artifact hashes are the transitive inventory, including
implementation identity and bound sources, manifest, evaluation reports,
verification, protocols and all nested raw reports. Packet writing copies that
inventory into an immutable SHA-addressed local CAS. Reconstruction does not
trust a saved derived body: it reloads preserved inputs and reruns validation,
evaluation ingestion, completeness, pane/path classification, aggregation and
the final decision. Before replay interprets any evidence, it rehashes every
regular, non-reparse CAS file against a completion-bound exact inventory.
Replay then resolves every manifest, report, protocol, schema, execution, raw
output, verification artifact and bound source strictly by digest from that CAS;
it never falls back to an absolute or relative external original. External
artifact paths therefore need not remain available and cannot mask a missing or
corrupt preserved blob.

`PASS` additionally requires complete applicable paired evaluations, all
required artifacts/edges, passing registered equivalence claims,
non-insufficient behavior and outcome classifications, and complete verification
evidence.
The ordered comparison surface must be nonempty and contain a present,
applicably paired boundary before either classification can report
`no-detected-change`. Optional not-applicable boundaries are skipped, but an
all-not-applicable surface is `insufficient/incompatible`. Every applicable
cohort must contain all mandatory behavior and outcome metrics for `PASS`.
Completeness is audited across the entire ordered surface before the first-change
classification, so an early detected change cannot hide a later partial cohort.
Every required edge must appear in `comparison_order`.

For each present comparable edge, pane and world, the packet retains source and
destination rows plus destination-minus-source differences and ratios. A zero
source denominator has an explicit `zero-denominator` status rather than an
invented number. Scalar outcomes and every registered service field receive
deterministic paired-world bootstrap intervals (1,000 resamples); fewer than two
paired worlds reports uncertainty as unavailable rather than a point interval.
The resampling unit is a paired world; summaries remain grouped by edge, pane,
endpoint node, arm and run, and optimizer/learner identities are never pooled.
Historical T3 seeds are explicitly unpaired, so world resampling does not
estimate optimizer uncertainty.

## Evidence and capability gaps

No production lineage manifest, reviewed evaluation/claim or equivalence
execution protocol, frozen verification protocol, historical
checkpoint inventory, migration/default records, semantic comparators,
evaluation corpus, evaluation rows, implementation identity, verification
artifact or review packet was collected by this implementation task. The known
historical contract-8 prior itself was not copied, changed or executed during
this implementation, and it still needs its checkpoint-specific reviewed adapter
protocol, default-resolution/migration record, historical worker contract or
reviewed producer comparator, and frozen equivalence corpus/receipts. No
production snapshot cohort or policy RNG schedule has been preregistered, and no
production service/missingness, trajectory or integral rows exist. The capability
tests and local bridge parity checks are diagnostic implementation evidence only;
they are not lineage evidence and make no selection/readiness claim.

The architecture requires its stated relative-throughput and incremental-RSS
budgets to be applied through preregistered profiles. This implementation
supplies bounded-capture receipts and the packet's structured benchmark evidence
path, but no production benchmark manifest has yet bound and executed the exact
scenarios, baselines, machine identity and measurement method against those
budgets. Total process working set alone is not accepted as incremental capture
cost. This is a remaining evidence/protocol `HOLD`, not an inferred performance
pass.

Therefore any packet built from the repository without those production inputs
remains `HOLD`. No VM was
started, no training or submission was launched, and no confirmation seed was
used.
