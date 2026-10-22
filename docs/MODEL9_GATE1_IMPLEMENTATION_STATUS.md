# Model9 Gate 1 implementation status

Status: the Gate 1 fail-closed validation framework is implemented for
independent Astra High re-review, but implementation-quality approval has not
been issued and production capabilities/evidence remain incomplete. Evidence
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

The current live evaluator is intentionally narrower than the architecture. It
can run a current-contract, strict-loaded deterministic checkpoint only for
`welfare`, `ending_residents`, `trade_utility`, `extinctions` and `settlements`.
Strict actor loading requires exact raw state-dict key coverage before PyTorch
load hooks run, so compatibility hooks cannot synthesize absent FiLM or region
tensors. Tensor-equivalence declarations resolve a top-level checkpoint mapping
plus ordinary dotted PyTorch state-dict keys and reject ambiguous paths.
It requires the exact checkpoint worker contract or complete reviewed producer
comparison over reset, configuration, action, observation, reward and state
trajectory semantics. The same historical-to-live check is recomputed for
imported reports, so a resealed report cannot bypass producer compatibility. It
does not silently substitute a fresh rollout or final snapshot statistic when
asked for an unsupported capability.

Review packets are stored under their canonical packet digest and contain CLI,
expanded worlds, artifact hashes, raw-row and aggregate hashes, equivalence
checks, verification evidence, missingness, supported/unsupported claims, separate behavior and
outcome classifications, a one-page decision, and a completion record binding
the output files. The implementation input is itself a content-addressed,
digest-sealed `gate1_implementation_identity` artifact. It records the clean Git
commit, exact source-file hashes and Git-porcelain dirty-state proof for the Gate
1 module, evaluator, focused test, this status file and both governing design
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
artifact or review packet was collected by this implementation task. In
particular, the repository still lacks a Gate 1
evaluator for:

- reset-snapshot input;
- seeded-stochastic action execution and a frozen policy RNG schedule;
- integrated population-years and QoL-years;
- canonical trajectory digests;
- complete service outcomes and compiler-intervention totals; and
- pinned legacy inference or a reviewed legacy adapter for any historical
  checkpoint that is not raw contract 12.

Therefore a full architecture-mandated Gate 1 request is rejected as unsupported,
and any packet built from the current repository alone remains `HOLD`. No VM was
started, no training or submission was launched, and no confirmation seed was
used.
