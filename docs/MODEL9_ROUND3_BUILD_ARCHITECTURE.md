# model9 round-3 build architecture

Status: **reviewed architecture and prospective protocol**, 22 October 2026. The original six-item architecture received
independent Sol and Astra review. Astra approved the strict teacher provenance, promotion, seed-role, curriculum,
calibration and experiment amendments on 22 October. This is not implementation or launch approval. No Round-3
experiment has been submitted from this build. Qualification work and final training submission are distinct gates.

This document reconciles the approved round-3 decisions in `MODEL9_M7.md`, `MODEL9_SEARCH_TEACHER.md`,
`MODEL9_PERSONALITY.md` and `MODEL9_TOKENS.md` with the six build items in the 22 October handoff. It is normative
for the implementation review. Where older prose differs, this document governs the round-3 build only.

The experiment freeze rule is strict: after a round is submitted, its code commits, simulator build, data, labels,
branch shards, flags, seeds and decision rules do not change. A defect found after launch creates a new named round.
Confirmation seeds 3000-3059 are denied in every seed-bearing input and command in this protocol. Any later release
requires separate authority and a new protocol; no qualification, inventory replay, comparison or smoke may use them.

## Scope and invariants

Round 3 adds:

1. the hierarchical region head, with an exact flat-head compatibility mode;
2. requested / accepted / realised diagnostics for every actor head through the native bridge;
3. a bootstrap for the constrained-policy cost critic;
4. source and trajectory provenance for search labels and critic branch shards;
5. calibration coverage by alternative kind and objective head; and
6. a flag-gated simulator request for raw material gathering, using the existing `materials` service head.

This work does not change the contract-12 actor observation or action widths. Diagnostics are sidecars, not actor
inputs. The materials request reuses the existing sixth goods-service head; it does not add a per-material action.
All compatibility flags and their resolved values must be written to run metadata.
The raw-material flag is **off throughout this experiment**. Its flag-on tests are implementation evidence only;
enabling it for training, search, qualification or evaluation is a separate treatment requiring new pre-registration.

## 1. Hierarchical region head

### Authoritative distribution

The actor owns one saved `region_head` mode: `hierarchical` or `flat`.

- `hierarchical` first chooses `native`, `stay`, or `go`, then chooses a legal destination inside `go`.
  `KEEP_INTENT` is a member of `go`, beside region rows other than the current-tile row.
- `flat` is the earlier masked softmax over native, keep-intent, stay and region rows.
- `region_logp` is the authoritative joint slot distribution used by sampling, behaviour likelihood, entropy, PPO,
  search-label imitation and stochastic evaluation. No consumer may reconstruct a different distribution from raw
  logits.
- Deterministic hierarchical inference chooses the group first and then the best legal member. Deterministic flat
  inference uses the earlier plain slot argmax. The old grouped-argmax evaluation remains a diagnostic only.

The group offset is zero-initialised. At zero offset the hierarchical joint probabilities equal the flat probabilities;
only deterministic selection differs. Illegal slots and empty groups have zero probability and finite gradients.

### Loading and resume

Head resolution happens once, after all actor and teacher state has been loaded:

1. an explicit CLI override;
2. the checkpoint's saved mode;
3. `hierarchical` for a checkpoint written before the mode existed.

`--region-head flat` is the supported exact-compatibility flag. An environment alias may be retained for old scripts,
but it must resolve through the same function and be logged. Every entry point that loads an actor (training, PPO,
evaluation, sweeps, search, movement audit and specialist/daughter tools) must use the same loader.

A pre-head optimiser snapshot must never be silently interpreted as a hierarchical resume. Exact old-run resume is
allowed only in flat mode, with dormant group parameters excluded from the actor optimiser so the saved parameter
groups still match. Starting hierarchical training from an old actor is a new run: load actor weights, initialise the
group offset to zero and create a fresh actor optimiser. If neither path is satisfied, fail with an actionable error.
This preserves the rule that an already launched run cannot change its action semantics.

An explicit override on resume must equal the setting frozen in that run. A different head mode creates a new named
run and may not reuse the old resume optimiser/RNG state. The same rule applies to the cost-bootstrap setting below.
“Exact flat compatibility” means the same actions, likelihoods, optimiser state and RNG continuation promised by the
old run. Constructing the dormant group layer must not advance the actor's global initialisation stream (save/restore
the RNG or use a no-RNG zero constructor), because otherwise later heads receive different initial values even when
the group is unused.

### Required tests

Test joint normalisation and masks, group-first mode, flat exactness, old actor loading, every public checkpoint loader,
explicit override precedence after resume, and both permitted optimiser paths. Search imitation must back-propagate to
both the group and destination terms for a movement label.

## 2. Requested, accepted and realised head diagnostics

`ElderCompiledDecision.Requested`, `.Accepted` and `.Realized` are the authoritative three stages. The native model9
bridge must expose all three directives for each controlled village decision; Python must not infer them from the
sampled action or from an intervention count.

The bridge payload also carries a stable per-head sidecar with validity masks so analysis can compare like units:

- retained field, retained herd, Development budget, field share and fishing share;
- six service levels in `ElderModel9Contract.GoodsServices` order;
- clothing and shelter routes;
- movement slot / intent;
- fission policy;
- commitment span/action;
- trade values, gates and export share.

`requested` is the policy's submitted value, including invalid values for diagnostic purposes. `accepted` is the value
after schema, legality, commitment and compiler repair. `realised` is the post-resolution control value reconciled by
`ElderDecisionRealization`. It is not a claim that an equal amount of goods was produced. A validity/activity mask must
mark native, absent and inactive heads instead of manufacturing a numeric outcome. Physical service diagnostics remain
separate: `ServiceRequested` and `ServiceGranted` are compiler units; any later realised receipt must come from an
authoritative resolver receipt, not from copying the grant.

Each sidecar names the trajectory identity, action decision turn and camp. The step returned after an action associates
the next observation with the preceding decision sidecar explicitly; consumers must not join them by array position.
An initial reset has no preceding decision and returns an absent sidecar. Fork/restore preserves the saved main-path
alignment, while each subsequent branch action receives its own trajectory identity.

The sidecar is returned by step/fork/restore and is recorded in replay, but is never copied into
`ElderObservation`, the actor tensor, reward or policy memory. PPO/evaluation aggregate per-head request-to-accept and
accept-to-realise change rates and magnitudes. Tests need at least one accepted action, one legality repair, one
capacity/resource repair and one post-resolution difference, with replay/fork equality.

## 3. Cost-critic bootstrap

The guard cost is a per-settlement-turn indicator, not a per-resident quantity. For every living sequence truncated at
the rollout boundary, its cost GAE bootstraps from the cost critic's value of the recorded next observation. Terminal
sequences bootstrap zero. `bootstrap_index` is the only mapping from final rows to next observations; world or row order
must not be assumed.

New runs default the bootstrap on. A resumed run keeps the setting saved in its own state. An explicit CLI flag is
resolved after resume metadata is read, logged, and must match that saved setting; changing it creates a new named run
rather than mutating a resume. Runs saved before the setting existed remain off when resumed, so their objective is
frozen. Tests cover reordered worlds, multiple tails, daughter settlements, no living tail,
termination, CLI precedence and resume persistence.

## 4. Label and branch provenance

Round-3 training accepts only attributable inputs by default.

Run-level provenance contains the full checkpoint SHA-256, a clean learner commit, simulator commit/build identity,
contract and utility versions, and a canonical digest of search configuration. VM production generation must use clean,
pushed commits. Local dirty smokes may record a dirty-tree digest, but `unknown`, a failed Git command, or a bare
`+dirty` marker is not sufficient for training provenance.

Every label row records seed, withheld mask, warm-up, decision turn, camp, personality and source-run identity. Every
branch records those fields plus alternative kind/slot, calibration flag, fork turn and, for daughter-bank starts, the
bank entry and snapshot-content hash. Directory names and branch identities include the full trajectory identity;
same-seed daughter starts must not overwrite or alias. Loaders rebase branch world IDs across all input roots rather
than trusting independently generated int64 IDs.

Policies are:

- `reject` (training default): fail on unknown sources, incompatible identities, collisions or an undeclared mix;
- `flag`: load an intentional analysis mix, warn, and persist the full source report;
- `allow`: tests and migrations only; still report sources.

Mixed-vintage labels are normalised per file before concatenation. Missing optional label targets get neutral masks;
missing optional observation sets get the contract's absent rows; missing personalities get the recorded default
personality. Intersecting keys across files is forbidden because it silently removes newer conditioning. Tests cover
known/unknown/mixed sources, dirty identity, duplicate trajectories, multiple bank entries, and legacy plus personality
labels.

### Native-teacher root provenance (reviewed amendment)

Production teacher shards must be generated through the exact clean production `Model9Bridge` binary used by this
protocol, not `Tools/Headless`'s separate assembly. The bridge is simulator-authoritative; a learner-side collector
only records its observations, native teacher labels and outcomes. The old t1 teacher bank is not eligible for
metadata migration: regenerate it. Old t2m is comparator-only, not an initialization, anchor or provenance parent.

Each generation source record contains full clean learner and simulator commits, bridge binary SHA-256 and schema
`buildHash`, exact rules/statics, native contract/action/observation/environment and label/outcome schema versions,
and canonical generation options. Options explicitly include the full seed list, withheld masks, warm-up policy,
rollout turns, players, movement-enabled state and opening-generation limit. Record the effective raw-material flag
as false; the actual rules/statics strings must match subsequent consumers, not merely be described as equivalent.
`source_run` is the SHA-256 of a versioned canonical source record excluding its own digest.

Each shard binds that source to its seed, mask, actual warm-up/opening turns, first decision turn, initial
fingerprint, configuration hash and full initial-snapshot SHA-256. External snapshot starts additionally record the
root family and full input-snapshot hash. A trajectory ID hashes the full attributable context, not only the seed.
Its payload manifest records every array's relative name, full file SHA-256, dtype, shape and row count. A canonical
dataset digest binds the sorted shard/trajectory IDs, metadata and payload manifests; neither directory names nor
MVIDs alone establish provenance. Publish metadata last, after payload completion; refuse inconsistent existing
output rather than overwrite it. Deterministic reruns verify the same payload/context or fail closed.

The production loader verifies source identities, actual binary/configuration compatibility, schemas, all payload
hashes/shapes/dtypes/row counts and duplicate trajectory/decision identities before normalization, normalizer fitting
or training. It rebases distinct trajectories to distinct sequence IDs even when their seeds/masks coincide; source
seed/mask/snapshot identity remains separate from those IDs. Reject legacy, unknown, dirty, mutated or incompatible
inputs. Multiple sources are allowed only by an exact frozen `source_manifest.source_runs` declaration and the
same compatibility rules as other inputs; an undeclared mix is rejected. Optional historical-field normalization
never upgrades unattributed data into production evidence. Train and dev source reports join labels and branch
sources in the pre-run compatibility check and frozen run manifest.

### Lineage schema 2 and deterministic promotion (reviewed amendment)

A random-initialized root is verified only with validated native-teacher source proofs for every training and dev
input, their payload/dataset digests, current exclusion manifest and actual role-family closures. Schema-1 seed lists
plus file hashes are insufficient. Rerun guarded clone optimizer seeds **0-9** on the regenerated bank under this
contract. Freeze the complete clone configuration, including input guard 10, schedule, optimizer, critic type and
budgets before fitting any clone. The candidate used for four-objective calibration must already contain all four
critic heads; a scalar-only clone is not sufficient. No calibration outcome may select a critic type or refit budget.

Exact current simulator/bridge identity is the default and is required for fresh Round-4 teacher lineage. The only
implemented comparator normalization v1 has `excluded_fields=[]`: decompress complete SaveCodec artifacts, retain
every header and binary-tail byte, and use the bridge's versioned canonical fingerprint-save export to prove the
complete decompressed WritePayload SHA equals the live native fingerprint (gzip artifact SHA is separate). Typed
reload validates root/turn, not the lossy restored-world fingerprint. Independently rehash these complete artifacts
and require byte-identical complete payloads across sides. Partial physical projections do not prove equivalence.
Any non-identity legacy/current normalization v2 requires separate architecture review; v1 grants no such waiver.

To avoid a circular content identity, protocol-digest projection version 1 canonicalizes every preregistration field
except its own `digest` and the explicitly late-bound `clone_protocol_sha256`. No other rule/configuration field is
excluded. Freeze this projected protocol, the exclusion manifest and teacher inputs, then the complete clone-plan
artifact; bind that plan's full SHA-256 into final preregistration bytes. A separate immutable Astra approval receipt
binds both the projected protocol digest and the SHA-256 of those complete final preregistration bytes. Approval is
external, not embedded in the preregistration. Before fitting, each clone verifies the projected identity, full frozen
plan and external approval; changing the late binding without a new matching approval fails. Promotion and final
qualification repeat the same checks. The clone-plan configuration projection is separately versioned and excludes
only optimizer seed/output paths and these explicitly late-bound protocol/approval references, not training choices.

A deterministic promotion utility validates every one of the ten candidate checkpoints and every associated
evaluation report, including failed/ineligible candidates. Reports bind the exact checkpoint, model/normalizer
identities, binary/configuration and complete per-world evidence. Derive selection families from report contents;
do not trust a user-entered seed range or a filename. Missing, duplicated, mismatched, nonfinite or incomplete
candidate/report evidence blocks promotion, rather than silently dropping a candidate.

The clone selector uses movement-on default-world total welfare divided by the paired heuristic total on seeds
2160-2179. Eligible ratios are in the inclusive interval [0.80, 1.20]; teacher action agreement is diagnostic only.
Choose the eligible candidate with the largest ratio; exact ties choose the lower optimizer seed, then the
lexicographically smaller full checkpoint SHA-256. If none is eligible, qualification fails. Freeze this selector and
tie-break before evaluation. Promotion records every candidate/report/input hash, all candidate lineage closures,
the selected init parent and the selection families actually inspected, not just the winning seed's evidence.

The promoted checkpoint is a **non-training, non-resumable artifact**. It carries no optimizer/RNG continuation that
could be interpreted as a run resume. Its actor, every critic and normalizer are bit-identical in tensor/value and
inference semantics to the winner; promotion may add provenance but may not repair a normalizer, change head mode,
reshape weights or consume an actor random stream. A new PPO run may initialize from its weights with fresh
optimizers. Recursive schema-2 verification checks both the winning parent and the entire selection closure,
including losing candidates and all inherited training/dev/selection families. A promotion wrapper cannot launder
an unverified ancestor. Required tests include each failure path, deterministic ties, report-derived family leaks,
transitive losing-candidate contamination, exact model/normalizer preservation and refusal to resume a promotion.

## 5. Calibration coverage and value range

Calibration inputs are held-out world and daughter-snapshot families excluded from critic training, actor training and
selection tuning. Their manifest and source identities are frozen with the run.

Calibration and coverage use the same definition of an evaluable branch. A branch is evaluable for an eight-turn
window when the lineage supplies eight realised turns or terminates earlier. A living lineage also needs a recorded
endpoint observation at turn eight for `V(s_8)`; eight reward rows without that observation are insufficient. A living
short tail is excluded from both calibration and coverage.
For each alternative kind and objective, report:

- evaluable calibration branches;
- branches with a non-zero realised window return; and
- calibration bias statistics when the cell has enough evaluable branches.

A zero return is a valid target and counts as evaluable, but ordinary relative bias is undefined. For objective `k`,
the frozen near-zero threshold is `tau_k = 0.01 * s_k_ref`, using
`contracts.OBJECTIVE_REFERENCE_SCALES`; a row is near zero when `abs(realised_k) <= tau_k`. Every cell reports
fixed-scale error `(prediction - realised) / s_k_ref`; non-near-zero rows additionally report relative error
`(prediction - realised) / realised`.

The near-zero and non-near-zero subsets are gated independently even when both occur in one cell. If the near-zero
subset is non-empty, its mean and median absolute fixed-scale error must each be at most 0.10 unless a non-mandatory,
pre-registered waiver names that exact kind/objective cell. Zero rows never disappear behind a cell-level relative
metric. For every non-mandatory kind/objective cell, a non-near-zero subset of zero rows makes the relative gate not
applicable; a subset of 1-19 rows is `insufficient_calibration` and fails; a subset of at least 20 must have absolute
mean and median relative error each at most 0.10. The non-zero count is reported separately to reveal an uninformative
cell. Mandatory cells retain the stricter explicit rule below.
Population bands (`0-29`, `30-99`, `100-299`, `300+`) and two-hot value-bin saturation are diagnostics, not substitutes
for kind/objective coverage.

Search deterministically forces one keyed calibration alternative for each kind in every world where that kind becomes
available until its per-world minimum is met; the ordinary keyed calibration share supplies the rest. It must not wait
past the only occurrence of a rare kind.

The calibration verdict cannot pass with missing mandatory kinds. Native, region and mix welfare cells are never
waivable: each must have at least 20 evaluable welfare branches **and** at least 20 non-near-zero welfare branches. On
those non-near-zero rows, the absolute value of both mean and median relative error must be at most 0.10; on any
near-zero rows, the fixed-scale gate above also applies. A mandatory cell with fewer than 20 non-near-zero welfare
branches reports `insufficient_calibration` and fails rather than treating an undefined relative statistic as a pass.
The report separately marks every kind/objective cell below 20 and every kind/population-band cell below 5. Each such
coverage mark is a failure unless pre-registered as waivable. This Round-3 protocol freezes `waivers: []` before
results: every coverage and error gate must pass, with no waived cells.
The generic nonmandatory waiver mechanism remains available to other separately pre-registered protocols, not this
one. A warning is never silently converted into `pass`.

Calibration is a one-shot pass/fail audit of the already promoted, hash-frozen clone candidate. Its critic and
normalizer are fixed before calibration results are read. Calibration cannot select among clone candidates, choose
an epoch/checkpoint, set a hyperparameter, cause critic refitting or trigger adaptive seed/snapshot/branch top-ups.
The same prohibition applies to changing a waiver after seeing coverage or errors. Failure blocks this protocol's
submission; a revised proposal requires a new name, fresh review and an explicit account of the exposed families.

## 6. Flag-gated raw-material request

The simulator-side design is `Docs/ElderModel9MaterialsRequestDesign.md`. Its key contract is:

- authoritative stock remains camp inventory plus `WorldState.MaterialSources`;
- requests, conditional authorizations, dedicated labour reservations and baseline receipt summaries are transient
  turn state;
- the request reuses goods-service index 5 (`materials`) and does not widen contract 12;
- the compatibility flag defaults off for old saves and baseline checks;
- Prepare records eligible known/discovered sources but never forecasts resolver-endpoint quantities;
- the ordinary baseline extraction and production path executes once against authoritative endpoint state; and
- after every camp's baseline extraction, production, optional goods settlement and needs, one global canonical pass
  may settle each authorization against captured baseline receipts, dedicated capacity and residual sources, before
  any demographic population mutation, entity removal, migration, fission or Close operation.

The raw request is additional stock-buffer gathering, not recipe input disguised as a second request. Prepare freezes
the ordinary target `B_m`, additional floor `F_m`, opening inventory `I_m`, request ceiling
`U_m=max(0,B_m+F_m-I_m)`, repaired service fraction `f`, and conditional authorization
`A_m<=f*U_m`. It reserves only genuinely unused prior-cohort intervals. Gatherer reservations are unavailable to food
and baseline gathering; specialist reservations are unavailable to baseline extraction and recipes. Absence from an
assignment list alone is not unused capacity.

The resolver captures inventory immediately before ordinary baseline extraction and its actual receipts. Late
settlement computes the remaining eligible deficit
`D_m=max(0,B_m+F_m-I_pre-baseline,m-R_baseline,m)` and commits at most
`min(A_m,f*D_m,dedicated_remaining_capacity,authoritative_residual_source)`, ordered by camp, material, role,
distance and stable source ID. It cannot revise a request, worker, source discovery, label, claim, ceiling or earlier
result. The receipt is next-turn inventory: production, trade and needs are not reopened. Using `PolicySurplus`
wholesale remains forbidden because it changes goods semantics.

The amended physical contract uses diagnostic schema 2. A provisional schema-1 replay produced under raw flag-on has
different settlement semantics and fails closed; it cannot be upgraded by relabelling diagnostics. Schema-1
continuation is permitted only when the raw flag was off and the legacy semantic fingerprint verifies.

Astra's focused ruling rejected Prepare-time endpoint-Wood prediction and the precomputed baseline-debit substitution.
The simulator architecture amendment above is the required resolution, subject to final cross-repository document
review before implementation. Passing a local build or the provisional implementation tests does not establish that
the simulator is green. The raw-material flag remains **off throughout Round 3**; flag-on support is implementation
evidence only and its conditional next-turn settlement is not a Round-3 treatment.

The final cross-repository workforce ruling makes this machinery **latent, not a completed useful full-turn
mechanic**. Current native fallback commits every residual worker, so correct normal Prepare authorizes zero raw
gathering capacity. Positive synthetic late-settlement tests validate the kernel only; they do not demonstrate a
positive normal-turn path. Useful positive gathering is deferred until a separate workforce-idleness contract is
reviewed and adopted. Round 3 remains raw OFF.

## 7. Seed roles and prior-use inventory (reviewed amendment)

The automated prior-use inventory is a prerequisite, not an assertion that an apparently fresh numeric range was
never used. It inventories teacher/label/branch sources, candidate ancestors, evaluation and promotion reports,
grassland/valley files, bank indexes and decompressed snapshot hashes; it derives actual root seed/snapshot closures
and fails on unavailable evidence or incompatible prior exposure. It never replays denied confirmation worlds.
Publish and hash the inventory before qualification. Any role change requires review before data generation.

| Role | Frozen roots / rule |
| --- | --- |
| Teacher TRAIN | 1000-1599, plus every explicitly enumerated training grassland, valley and daughter root/snapshot |
| Teacher DEV / clone fitting selection | 2100-2159 |
| Clone promotion selection | 2160-2179 |
| PPO checkpoint selection | 2060-2099 |
| Development reports and personality sweeps | 2000-2059 |
| Calibration | 10000-10099 and only daughter snapshots descended from these roots |
| Confirmation | 3000-3059: denied everywhere in this protocol |

Manifest seed arrays contain expanded integers, not range strings. All exact TRAIN fixture/root lists, withheld
masks and warm-ups are frozen and hash-bound; a path to a mutable bank is insufficient. Calibration daughter
snapshots are captured by a fixed non-training procedure from calibration roots only, deduplicated by full snapshot
SHA-256, and frozen **before any of the ten clone fits**. Their count and selection rule are fixed without inspecting
calibration errors. The training snapshot set and calibration snapshot set have no shared content or root families.
The legacy daughter `% 10` partition is not proof of calibration holdout: a root used for ordinary TRAIN remains
ineligible for calibration even if one of its snapshots was previously called held out.

The calibration manifest binds calibration roots/snapshots, declared training/selection exclusions and actual role
closures; the role lists are not exhaustive positive allowlists. Exact intentional source mixes are controlled by
`source_manifest.source_runs`. All calibration-family intersections with training, dev, promotion or PPO selection,
including inherited losing-candidate closures, are rejected. Development reporting cannot cause reselection or
retuning of a frozen candidate. Focal reports must contain complete per-world focal evidence, not only pooled totals.

## 8. Curriculum and matched Round-3 arms (reviewed amendment)

### Deterministic `r3-balanced-v1` curriculum

New Round-3 mixed-use runs use this ordered 12-slot cycle; the names denote source kinds, not economic regimes:

| Slot | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Source | default | daughter | grassland | valley | fixture | default | daughter | grassland | valley | fixture | default | daughter |

`default` means a fresh ordinary TRAIN opening and is mandatory: no optional stream replaces slots 0, 5 or 10.
The cycle has three defaults, three daughters, two grassland, two valley and two capability fixtures. A missing or
ineligible daughter/grassland/valley source falls back to a fresh default, without borrowing a mandatory default
slot. Each realized fixture advances a separate rotating mask cursor, in herding/farming/fishing order. Availability
checks and empty-bank fallback do not consume an otherwise absent random draw.

A global slot cursor advances across draws, including partial cycles; save/restore both it and the capability-mask
cursor with the NumPy RNG. Freeze the schedule name/version, ordered slots, stage, fallback rules, fixture order and
initial cursor values in the run manifest. Log requested and realized source kind and counts, source root/mask/
warm-up/snapshot, separately from economic regime. A banked mask-zero world is not counted as a fresh default.
Legacy resumes retain their frozen legacy schedule; changing schedules starts a new named run, not a resume.

Each arm/optimizer-seed run receives a private bank initialized from an identical frozen bank image: index plus
snapshot-content manifest. Captures and evictions thereafter belong only to that run; no arm reads another arm's
evolving bank. Freeze capacity, reservoir/capture rules, eligibility, initial image and root/snapshot closure. Tests
cover all populated streams, each fallback, partial/repeated draws, positive fresh-default allocation, rotating
fixtures, empty-bank RNG equality, private banks and exact resume continuation.

### Arms and fixed budget

All three arms use paired optimizer seeds **170-174**, the same promoted guarded clone and identical non-treatment
settings. Each seed identifies a matched triplet; do not bootstrap the three arms' seed indices independently.

| Setting | R3def | R3per | R3lo |
| --- | --- | --- | --- |
| Personality | default [1,0,0,0] | Dirichlet(1) | Dirichlet(1) |
| Teacher KL | 0.01 | 0.01 | 0.002 |
| Training search cohort | default | shared Dirichlet(1) | the identical shared Dirichlet(1) cohort |

Common settings: token actor, hierarchical region head, mixed-use stage, learned movement/fission/trade, experts,
four-head vector critic for every arm, cost bootstrap on, reference objective scales, normalized-component clip 5,
daughter drift concentration 200, branch share 0.25, KL refit on and FiLM bound off. Capital commitments remain off;
single-turn behavior is explicit. Use `r3-balanced-v1`, 12 worlds per iteration and 25 turns per world, **five
critic-only iterations followed by 50 actor iterations** for every run, regardless of whether heads are newly
created. Warm-up counts overlap by maximum, not addition, and do not repeat after a completed warm-up on resume.
The raw-material simulator flag remains off in every arm and every associated search/evaluation.

Freeze all resolved values in the manifest, not just CLI overrides: actor/critic architecture, optimizer/LRs, epochs,
minibatch sizes, clipping, entropy, target KL, cost budget/multiplier, shaping, anchors, reference scales, search
weight/margin, bank and curriculum policies, rollout/selection counts, every RNG seed, rule/static string and input
identity. In particular `--learned-trade` can add default trade credit: explicitly freeze the base objective vector
to welfare=1, population_years=0, trade_utility=0, qol_years=0, so the default personality remains welfare-only and
Dirichlet personalities alone supply their registered preference weights. Record this override rather than infer it
from the word `default`. No unlisted difference between
arms, legacy default or checkpoint inheritance may silently change the treatment.

### Frozen search cohorts and one-shot calibration generation

Generate two training cohorts from the promoted candidate: one at the default personality for R3def, and one at
Dirichlet(1) shared byte-for-byte by R3per/R3lo. Each uses **60 TRAIN openings plus 60 frozen TRAIN daughter
snapshots**, with exact same world/snapshot lists between cohorts; only the registered personality/scoring context
differs. Freeze main paths of 25 turns, first searched turn one, four decision points, two villages per point,
horizon six, policy tail four, critic bootstrap off,
daughter search on and every current alternative kind (policy/native/region/mix/fission/trade). Use utility version 2,
fixed reference scales, imitation margin 0.004 and weight 0.5 before generation. All training tails are
four turns: training-cohort calibration share and forced-calibration minimum are zero. These are critic training
inputs, not held-out calibration evidence.

Separately generate calibration-only search once from the already promoted candidate, on **100 roots 10000-10099
plus the pre-frozen set of at most 100 calibration daughter snapshots**. Freeze the actual snapshot count/list,
personality/scoring context and capture selector before clone fitting; calibration uses 25-turn main paths, first
searched turn one, four points, two villages,
horizon six, bootstrap off, tail twelve, calibration share one and a forced minimum of one available
alternative per kind per world. Rare kinds are forced at their
first availability, not after it has passed. Every produced calibration label, branch and family is excluded from
actor/critic fitting and selection. No adaptive top-up is allowed on observed errors or insufficient coverage.

The current `MAX_ALTERNATIVES` is **19**: policy + native + seven non-current region rows + three mix vertices + two
fission probes + five trade probes. Freeze this count and the actual alternative definitions; legality, duplicate
policy choices and trade activity can reduce realized counts. A changed alternative set requires a new protocol.
One forced branch does not certify global coverage: all of section 5 must still pass with `waivers: []`.

## 9. Formal decisions and compute envelope (reviewed amendment)

### Personality separation and cost

Use the frozen seven-point personality grid and development worlds 2000-2059, for both all-village and focal sweeps,
on both `best.pt` and `latest.pt`. Preserve per-world QoL, residents, welfare, extinctions and the focal lineage/
population denominators needed to reconstruct each focal metric. Missing focal per-world evidence fails the
measurement gate. Selection of `best.pt` uses only 2060-2099 under the frozen selection rule; `latest.pt` is the
fixed 50-actor-iteration endpoint. A missing selected best checkpoint fails that cell; do not substitute latest.

For each of R3per and R3lo, each checkpoint form and each sweep form, let `q_i` be the mean across the same worlds
of `QoL(qol personality) - QoL(population personality)` for optimizer seed `i`. Let `p_i` be the corresponding mean
of `residents(population personality) - residents(qol personality)`. Use focal quantities in focal mode. Compute the
across-five-seed sample SD of each difference separately (`ddof=1`). Seed `i` succeeds only when both `q_i > SD(q)`
and `p_i > SD(p)`, strictly. Require at least four of the five seeds to succeed in **each of the four cells**
best/latest x all/focal, separately for both personality arms. Teacher agreement is not part of this rule.

At the default personality, compare R3per/R3def and R3lo/R3def for both checkpoint forms. The welfare ratio is the
ratio of arm totals on the paired worlds, pooled over the five matched optimizer seeds, not the mean of per-world
ratios. Its **entire paired 95% CI** must lie within the inclusive interval [0.80, 1.20]. Report paired R3lo/R3per
with the same estimator beside these gates. Freeze 4,000 bootstrap draws, a bootstrap RNG seed of zero and percentile
endpoints 2.5%/97.5%; each draw resamples matched optimizer-seed indices jointly across arms and the same root-world
indices jointly, retaining fixture/focal associations. Independent arm-seed resampling is not the paired rule.

For the default-personality R3per-minus-R3def and R3lo-minus-R3def extinction differences, on both checkpoint forms
and each registered reporting regime, the paired 95% CI upper endpoint must be <= 0. Report R3lo-minus-R3per too.
The global no-collapse evidence also compares each arm with the frozen promoted reference on the same worlds and
evaluated personality points; record the comparison/denominator explicitly and require extinction-difference upper
CI <= 0. Settlement count and below-30 share remain diagnostic unless the pre-launch movement audit judges the
behavior degenerate; they are not silently converted back into a 2% hard gate. Report population/QoL splits and
specialist share so welfare alone cannot hide a degenerate frontier.

### Compute accounting (not a runtime or price estimate)

The execution tier is part of the protocol. Local training runs are validation/stability probes only and use at most
three seeds. They cannot select a checkpoint, arm, hyperparameter or protocol and cannot establish readiness. Any
readiness-bearing training comparison runs on the VM with no fewer arms or optimizer seeds than T3: the three matched
R3def/R3per/R3lo arms and five paired optimizer seeds per arm (15 runs). The same seed identities remain paired across
arms. Reducing that VM cohort requires a newly reviewed pre-registration, never an inference from the local sample.
This boundary prevents a convenient three-seed local result from becoming a biased selection screen.
Every protocol-bearing train/PPO/clone-selection entrypoint enforces this boundary before fitting. Qualifying VM
plans bind the exact queue phase/argv and actual host project/instance/zone metadata; local probes require an explicit
frozen nonqualifying design and cannot emit selection, formal-completion or qualifying-lineage artifacts.

Before any native Reset, warm-up, branch or action, actual worker identity and read-only snapshot metadata must match
immutable producer/world contracts (binary SHA/MVID, full simulator source/build, schema versions, actual flags,
raw-OFF features, root/turn/snapshot/cohort). Parent aggregation rechecks returned proofs before optimization or
selection, and evidence retains them. Formal reporting binds every completion/checkpoint to one qualified exact-15
submission, and invokes the same shared non-training semantic qualification validator used by preflight; immutable
hashes or a `status: pass` shell alone never establish truth. Expensive training/evaluation is not rerun.

The 15 PPO runs have a nominal collection ceiling of
`3 arms * 5 seeds * (5 critic-only + 50 actor) * 12 worlds * 25 turns = 247,500`
post-reset world-turn resolves: 22,500 critic-only and 225,000 actor-training resolves. Earlier termination can
reduce actual collection. This is not a village-decision
count and excludes opening generation, fixture warm-ups, evaluation and bank-capture/serialization overhead. KL refit
reuses collected data and adds critic compute, not new world turns.

Let `A=19`, `P=4`, `V=2`, `H=6`. The two training cohorts have at most
`2 * (60+60) * P * V * A = 36,480` branches and `36,480 * (H+4) = 364,800` branch world-turn resolves, plus
their main-path rollouts. If those are 25 turns, the main paths add 6,000 resolves. For `N_c <= 100` frozen
calibration snapshots, the one-shot audit has at most `(100+N_c) * P * V * A` branches and
`(100+N_c) * P * V * A * (H+12)` branch resolves; at `N_c=100` these are 30,400 branches and 547,200 resolves,
plus at most 5,000 main-path resolves at 25 turns. Termination, legality and deduplication reduce these upper bounds.

Teacher collection costs `(N_train + N_dev) * teacher_turns` post-reset resolves plus opening/warm-up work; use the
actual expanded fixture lists, not just 600+60 if additional teacher fixtures are generated. Ten clone fits share
that dataset. Clone actor batches total `10 * epochs * ceil(teacher_rows / batch_size)`, plus the frozen critic
pretraining rounds/epochs. Record the exact budgets before fitting; this formula does not supply missing settings.

Personality sweeps alone have a nominal ceiling of `2 arms * 5 seeds * 2 checkpoint forms * 2 sweep forms * 7 grid points * 60 worlds
* 25 turns = 420,000` world-turn resolves. Add default-personality regime reports, promotion/clone evaluation,
movement controls/forks, daughter evaluation, replay, tests and retries as separately itemized costs. Runtime, RAM,
currency cost and Spot-retry overhead are **unknown until measured**; none of these formulas authorizes VM spending.

## Validation and launch gate

Before merge, run the focused round-3 Python tests, the specified movement/clone/autonomous suite, bridge smokes on
TRAIN seeds only, and simulator headless checks with the materials flag both off and on. The new optional simulator
configuration/receipt fields legitimately change serialized config/build fingerprints even when the flag is off, so the
flag-off comparator requires exact semantic world outcomes and material-transfer amounts against the prior branch, then
records a new immutable fingerprint baseline. The flat-head learner comparator still requires exact logits, sampled
actions, optimizer continuation and RNG trajectory against its pre-hierarchy baseline.

The pre-launch movement audit from `MODEL9_M7.md` is also mandatory: moves per settlement split by settled/nomadic
status, fixed capital abandoned per move, and welfare decomposed between fresh-stock harvest and steady state. The
pre-registration records an explicit judgement on whether the near-every-turn movement and small-settlement share are
degenerate; unit tests alone cannot discharge this gate.

Use separate gates and receipts; do not collapse a passed unit test or qualification phase into submission authority:

1. **Protocol gate:** Astra approved this amendment, including the focused endpoint-Wood/new-field-capital ruling.
   Freeze hypotheses, selector/tie-break, complete resolved clone/search/arm/evaluation configurations, all seed
   roles, snapshot-capture selectors, statistical rules, `waivers: []` and spending/resource envelope before relevant
   results are observed. The architecture is reviewed, but the implementation is not yet simulator-green or launch
   approved.
2. **Source gate:** automated prior-use inventory and denylist checks pass. Both repositories and the exact
   production bridge binary are attributable to clean pushed commits. Regenerate teacher TRAIN/DEV through that
   bridge; validate all source/payload/dataset proofs. Freeze calibration daughter snapshots before clone training.
3. **Clone gate:** fit all ten guarded clones, validate every selection report, and run deterministic schema-2
   promotion on 2160-2179. The selected movement-on welfare/heuristic ratio is in [0.80, 1.20]. Freeze all candidate
   closures and the promoted artifact. Old t2m is comparator-only; no teacher-matching gate or old-bank migration.
   Record historical t2/t2lo best/latest and all/focal results, T3lo/T3per and both/T3def comparisons, with missing
   evidence labelled insufficient rather than reconstructed or assumed to pass.
4. **Search gate:** the two training cohorts, exact opening/snapshot lists and all flags/alternatives are frozen;
   validate labels, branches, source manifests and compatibility with the actual production binary. R3per/R3lo
   share the same immutable Dirichlet cohort; calibration inputs never enter either training cohort.
5. **Calibration gate:** audit the already promoted candidate once on the frozen held-out generation. All numerical
   and coverage rules pass with no waivers. No selection, tuning, refitting or adaptive top-up uses the result.
6. **Movement/integration gate:** movement-on promoted-clone controls/forks produce the required settled/nomadic,
   abandoned-capital and fresh/steady welfare evidence and an explicit nondegeneracy judgment. Verify requested/
   accepted/realised head and physical sidecar integration, replay/fork/restore, cost endpoints/resume, flat exactness,
   private-bank curriculum and promotion/source/denylist regressions. Simulator semantics remain blocked until the
   focused cross-repository ruling and required comparator evidence are approved.
7. **Submission gate:** after qualification, freeze the final reviewed clean pushed commits, actual bridge
   SHA-256/buildHash, promoted init, manifests, full expanded seed/snapshot lists, queue job and commands. Preflight
   hash-verifies comparator evidence and every qualification queue-phase receipt, including output hashes and done
   markers; receipts must match the named code/configuration/input identities. Verify denylisted seeds across all
   commands/manifests/ancestors and all binaries, not just the PPO seed range. A success marker without verified
   complete outputs is insufficient. Do not mutate an existing job or reuse a receipt after changing its identity.

Local validation for these gates is capped at three seeds and supplies stability evidence only. Every training-workload
readiness verdict uses the VM's full three-arm, five-paired-seed cohort; local probe outcomes never filter which VM arms,
seeds or checkpoints are run.

Qualification may run before the final Round-3 training freeze under an approved qualification protocol; it is not
permission to start the 15 experiment runs. The final launch uses `elderq` / `tools/vmq.ps1` from the immutable
package only after all gates pass and explicit submission authority exists. Every queue phase is idempotent, writes
its completion marker last and resumes only its own frozen state. No code/data/flag/seed/decision-rule changes occur
after submission. Stop the VM when authorized queued work and result collection are complete. Runtime/cost remains
unknown until measured, and this document itself authorizes no VM start or submission.
