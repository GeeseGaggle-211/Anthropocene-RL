# Model9 economic observability and two-objective personality architecture

Status: Astra High approved for diagnostic-only implementation, 22 October 2026. This document is normative for the diagnostic build. The two-objective learner in section 8 is an experiment proposal and is not authorized for implementation or training until diagnostic evidence and a second architecture review exist.

## 1. Question and decision boundary

T3's held-out welfare regression tells us that the learner failed, but not which economic control failed, where a request was constrained, what production actually occurred, or whether the critic assigned the wrong value before the actor update. The diagnostic build must join, per village decision:

1. the actor's distribution and sampled request;
2. the simulator's requested, accepted and realised controls;
3. the physical cultivation, herding, manufacturing and trade ledgers;
4. population and QoL outcomes; and
5. the critic prediction, bootstrap, target, TD residual and actor advantage.

Diagnostics do not change the policy observation, reward, action, authoritative world, replay fingerprint, seed stream or promotion rule. Welfare remains an evaluation outcome. No existing checkpoint is promoted by this work.

## 2. Current evidence and observability gap

- The model9 bridge already returns village welfare, population, QoL, trade utility, specialist share and summed goods output.
- Diagnostic envelope 2 records per-head requested/accepted/realised values and schema-2 raw-material request/grant/receipt triples.
- The simulator already owns authoritative turn receipts: cultivation and other food transfers, managed-herd reports, production receipts and limiting diagnostics, material receipts, demographic reports and settled trade records.
- PPO already computes critic values, bootstraps, lambda returns, objective advantages and guard-cost targets, but these are reduced to iteration summaries and are not joined to the physical turn.
- Existing branch calibration found a strongly negative near-term critic bias on a tiny sample: growing lineages were valued much more highly eight turns later than at the branch entry. That is a warning, not a diagnosis.

The missing object is a stable join over `(run identity, arm, optimizer seed, world seed, trajectory, turn, camp, society)`.

## 3. Simulator diagnostic contract

### 3.1 Envelope and lifecycle

Diagnostic envelope 3 adds `economicDiagnostics`, keyed by camp ID. Each record has its own schema version 1, transition ID, trajectory ID, decision turn and camp ID. Reset uses envelope 0 and carries no preceding receipts. A restored observation may expose the preceding path's immutable sidecar for inspection, but is labelled `restoredObservation=true`, retains its original transition ID and is never emitted as a new transition row. The first step after a fork receives a new trajectory ID plus `forkId`, `branchId` and `parentTransitionId`; it cannot reuse the restored transition ID.

Replay verification requires the complete environment context: canonical reset snapshot and configuration hashes, producer binary/contract hashes, action sequence, background-policy sequence, RNG/state fingerprints and the expected transition graph. Paired continuation tests restore once and run two independently named branches; independent replay starts from the bound reset state. Both compare action, observation, reward, state fingerprint and economic sidecar. A sidecar mismatch is a replay failure, never “unverifiable”. Deep-copy tests mutate one fork and prove no alias in the other, including extinct camps and newly founded camps.

The sidecar is reporting-only. It must never be copied into `ElderObservation`, the 512-wide legacy tensor, token rows, teacher labels, reward vectors or authoritative saves. The bridge advertises the supported envelope and economic schema in `hello`.

### 3.2 Common outcome fields

Each village record carries the step duration; decision-opening resident and linked population; `DemographicReport.StartingPopulation` and `EndingPopulation` (both are linked population, including source-linked hunter parties); resident and party snapshots immediately before demography, after economy and after reintegration; gross births and deaths; cause-specific deaths where present; fission `SplitOff`; emigrated and immigrated residents; dispatch and reintegration headcounts; committed ending resident and linked population; ending QoL, nutrition, clothing and housing; food required/produced/consumed; opening and ending reserves; and the demographic limiting factor. Values come from the committed `TurnRecord`, executed movement orders and committed camp/hunter state, never from reconstructed observations.

The population boundaries are not interchangeable. The report must expose and test these equations in execution order:

`report starting linked = resident before demography + source-linked parties before demography`

`report ending linked = report starting linked + linked gross births - linked gross deaths + integer carry/correction`

`post-reintegration linked = report ending linked`

`committed linked ending = post-reintegration linked - split off - emigrated + immigrated`

Dispatch before economy and reintegration after economy transfer people between resident and hunter-party snapshots, but do not enter either linked-population equation. A separate resident distribution equation uses the captured pre/post resident and per-party values; it must not reinterpret the linked demographic report as resident population.

Every non-zero remainder is emitted as a named signed reconciliation residual. Daughter creation, migration, dispatch/reintegration and their associated food/inventory/capital transfers are explicit typed transfer witnesses; they are never folded into production, consumption or unexplained loss. A report with a non-zero unexplained population or stock residual is incomplete rather than approximately conserved.

### 3.3 Cultivation ledger

The cultivation block records:

- capability/activity and whether the head was active;
- requested, accepted and realised retained-field operation;
- requested, accepted and resolved field-development people;
- assigned farmer-years, maintenance worker-years, expansion worker-years, farm-tool-covered worker-years and gathering-bridge worker-years;
- opening, mean and ending field capital, capital added and capital lost;
- cultivated food delivered and cultivated food per farmer-year;
- the planner's field marginal return, selected field budget, final field budget, useful service, priced work and unmet service; and
- the provisional-versus-realised gathering bridge correction.

Field capital totals are captured at the authoritative horizon allocation, not inferred after resolution. Food is summed only from committed cultivation transfers for that camp.

### 3.4 Herding ledger

The herding block records:

- requested, accepted and realised retained-herd fraction;
- requested, accepted and resolved herd-development people;
- planned and covered herder-years and herding-tool use;
- starting, surviving, newborn, starvation-death, slaughtered and ending animals;
- animal-weighted food nutrition, water nutrition, total nutrition and management fulfilment;
- potential and accepted animal food, human food displaced, herd food delivered and slaughter food; and
- the planner's herd marginal return, selected/final herd budgets, target and development animals.

No animal can be counted twice when a camp has multiple managed groups. Ratios use explicit denominators and report them.

### 3.5 Manufacturing ledger

The manufacturing block contains:

- a compiler-control funnel in the stable order `food, clothing, shelter, tools, storage, materials`: requested and granted control quantities, their per-index unit tags and the action level that capped them;
- one row for every compiler request: stable request identity, service, service class, request kind, `CraftGood`/raw material/role where applicable, native request, post-action request, grant, unmet reason, native unit tag, worker-years equivalent and every conversion rate used;
- one physical row for every stable `CraftGood`: useful demand, planned output ceiling, realised output in that good's own units, worker intervals/years, equipment-covered work, and consumed Stone, Clay, Wood, plant food, animal food, animal fabric, cloth, housing material and tools;
- one physical row for Stone, Clay and Wood extraction: material units requested, authorized and received, assigned worker intervals/years, extraction rate, eligible-source count and receipt shortfall;
- the physical production-limit fields when an attempt was constrained: labour/tool, material, food, dependency and storage ceilings; and
- route selections and manufacturing-food debit.

The compiler arrays are not a physical-goods ledger and are never summed across services. Their declared units are: food worker headcount over the stated horizon; selected-route clothing output units; selected-route shelter output units; mixed tool-good units; pot units; and raw-extraction worker-year equivalents. In particular, `FinalizeRawMaterialPhase` converts material units with `ServiceEquivalentPerUnit = 1 / annualRate`, while the individual requests and receipts remain physical material units. Clothing, shelter and tool totals can also mix routes or goods, so they support only within-service request-to-grant control ratios. Physical grant-to-realisation ratios are computed only within a stable good/material row with matching units and explicit denominators. There is no synthetic "realised service" scalar.

Instrumentation snapshots each request before the service-level loop mutates `Request.Units`, then records five distinct phases: native demand, actor-capped request, compiler grant, executed worker/input claim, and committed physical receipt. For goods shared by multiple requests, current-turn receipts are attributed deterministically by `(service class, original request ordinal)` up to each granted good quantity; opening-stock draws, reserved intermediates, exports and unallocated surplus remain separate fields. Recipe and route coefficients are copied from the bound configuration into the record, so final-service fulfilment is the allocated final-good receipt plus an authoritative stock draw after subtracting intermediate consumption. This diagnostic attribution never feeds resolution.

An inactive action head has `active=false` and null request/accept/realisation values. An active head with no native demand has numeric zero phase values, denominator zero and `ratioStatus="no-demand"`; its ratios are null. A constrained positive request has its positive denominator, categorical constraint reasons and separately named loss at each phase.

### 3.6 Trade ledger

The trade block records eligibility and the turn's requisition/honoured/countered/ignored counts; number of settled trades; mean distance; made-to-order count; requested/opened export-labour share; delivered fractions; realised utility; food paid/received; and, for every stable `TradeGood`, imported and exported quantities. Detail rows retain stable transaction and leg IDs, requester/responder endpoints, requested/agreed/delivered quantities, stock/made-to-order split, escrow/claim, storage rejection, refund/return and both planned and realised endpoint utilities. Aggregates are derived from those rows at camp and world scope. Planned gains and physically realised utilities remain separate. Partner identities are diagnostic metadata and never policy inputs.

### 3.7 Cost, ownership and scaling

Receipts, herd reports, transfers and trades are indexed by camp/transaction once per resolved turn. With `R` physical receipts/transfers, `H` managed-herd reports, `T` trade records, `V` active villages and `G` stable goods/services, total capture work is `O(R + H + T + V*G)` plus serialization, not `O(V*(R+H+T))`. Stable arrays are bounded (11 craft goods, 11 trade goods, 6 services). No tile scan, historical scan or sub-turn loop is introduced.

Capture configuration is environment-owned and passed through preparation/resolution. It is not a mutable process-wide flag. If an existing static diagnostic hook must temporarily be used during migration, a disposable scope restores it in `finally`, the worker enforces one environment per process and a concurrency test proves that a second environment cannot observe it.

The simulator retains only the current envelope and one fork copy: `O(V*G + R + H + T)` objects, with no historical accumulation. The learner writer uses a bounded queue of at most two rollout batches and 256 MiB; full queues apply backpressure and never drop rows. Measured acceptance budgets are: diagnostic-off fingerprint identity and no more than 2% median throughput regression; diagnostic-on no more than 15% median step-time regression on the current model9 fixture, no more than 20% on the mature-population fixture, at least 80% of baseline end-to-end RL transitions/second with eight workers, and no more than 256 MiB additional peak worker RSS. The benchmark manifest, machine identity and raw timings are report artifacts; failure blocks VM collection.

### 3.8 Conservation boundaries

Each physical ledger declares an opening boundary (the decision snapshot), an interval event boundary and a committed closing boundary. The diagnostics emit the terms and signed residual for these equations, per stable item and then at world scope:

- residents: decision-opening resident/party distribution, dispatch before demography, linked births/deaths plus integer carry, reintegration after demography, then split-off/emigration/immigration to committed closing residents; dispatch/reintegration are internal linked-population transfers;
- food: opening plant/animal/reserve stocks + actual harvest/slaughter/gathering/trade/migration/fission inflows - meals - growth food - manufacturing feedstock - herd food - actual trade/migration/fission outflows - actual discard/decay = closing stocks;
- Stone, Clay and Wood: opening stock + extraction + trade/transfer inflows - recipe use - trade/transfer outflows - discard = closing stock;
- each `CraftGood`: opening stock + actual final/intermediate production + actual accepted imports/transfers - recipe consumption - service consumption/depreciation - actual exports/transfers - actual discard/destruction = closing stock;
- each herd ID: opening animals + births + ownership inflows - biological/starvation deaths - slaughter - ownership outflows = closing animals. Split/merge records carry parent and child herd IDs, never positional attribution;
- field capital by field ID: opening capital + constructed capital + transfer inflow - decay/abandonment/transfer outflow = closing capital; and
- each trade transaction: both endpoint IDs, both good legs, requested/agreed quantities, stock versus made-to-order supply, escrow/claim, delivered/accepted quantities, storage-blocked potential, refund/return and realised endpoint utilities. Only actual debits and accepted credits enter stock conservation. A request, undelivered agreement or quantity cut by `RoomUnits/Limit` is a funnel shortfall, not an inventory loss. Sender actual outflow equals receiver actual inflow plus an explicitly recorded actual return/destruction term. World-scope sums reconcile actual imports and exports per good.

Intermediate output consumed by another recipe is reported as intermediate throughput and never counted again as fulfilled final service. Production/storage limits and trade rejections remain potential-versus-actual funnel losses; because rejected potential never enters inventory, it is never subtracted in a stock equation. Missing authoritative terms are `null` with an availability reason, not zero. Any unexplained non-zero population, stock, herd, field or bilateral-trade residual marks that slice incomplete.

## 4. Learner trace contract

An opt-in `--economic-trace <directory>` writes a content-addressed rollout artifact. `transitions.jsonl` has one canonical row per village decision; `updates.jsonl` has one row per optimizer minibatch/epoch plus an iteration summary. Each transition binds:

- transition ID and successor transition/lineage IDs; run, rollout, worker/environment slot, arm, optimizer seed, world seed, trajectory, turn, camp, society and policy version;
- pre/post canonical state fingerprints, snapshot hash, full configuration hash, simulator binary hash, bridge/worker contract hashes, actor/critic/cost-checkpoint hashes and observation-normalizer hash;
- elapsed years, per-row `gamma = exp(-years / horizonYears)`, horizon-years and lambda; termination versus truncation, reason, bootstrap eligibility, destination observation/lineage ID and bootstrap checkpoint/version;
- raw simulator objective outcomes, explicit shaping debits, shaped rewards that enter GAE, objective reference scales, value-head units, population scaling and normalizer moments;

- actor means/log-stds or logits, sampled action, head log-probability, entropy and active masks;
- requested/accepted/realised action sidecars and interventions;
- the simulator economic sidecar;
- per-objective critic prediction at `s_t`, bootstrap prediction at its declared destination, raw TD residual, GAE/lambda target, raw advantage, reference-scaled advantage and objective-clipped advantage;
- the final pre-update policy advantage after personality/objective combination, standardisation, guard-cost subtraction and regime balancing, together with every objective weight, personality weight, balance weight, cost advantage and Lagrange multiplier;
- cost-critic prediction, bootstrap, target and residual; and
- personality, capability/regime mask and explicit missing/unavailable reasons.

`updates.jsonl` is written after the pre-update transition rows and around each actual mutation. It binds the exact batch-row digest and before/after policy versions, then records PPO ratio and clip fraction (distinct from objective-advantage clipping), approximate KL and entropy by active action head, policy/value/cost losses, teacher/auxiliary losses, per-objective critic loss, actor/critic/cost gradient norms before clipping, clipping scales, parameter-update norms and fixed-audit-minibatch value/action change. Metrics that are not computed are explicitly unavailable; they are not reconstructed later.

Transition rows are written after targets and final PPO advantages are computed but before optimizer mutation. A failed or partial trace is never silently accepted: the aggregator requires done markers with row counts, file SHA-256 values and the ordered transition/update Merkle root. This is a prospective training/update trace; a final checkpoint evaluation can produce current-policy predictions and forward realised returns, but must not label them as the historical predictions, lambda targets, PPO advantages, ratios or gradients that drove an earlier optimizer update.

## 5. Aggregator and reports

`python -m anthropocene_rl.autonomous_v7.economic_diagnostics` consumes one or more complete traces and writes deterministic JSON, CSV and Markdown without adding a runtime dependency. Its outputs are:

- `field_dictionary.json`: canonical field order, source object/phase, unit, sign/range, denominator, missingness and schema;
- `decision_rows.csv`: the flattened join;
- `domain_summary.csv`: requested-to-granted, granted-to-realised and productivity funnels by arm/domain/regime;
- `critic_calibration.csv`: optimization-target fit and independent-return calibration as separate target classes, with count, target/prediction means, signed bias, MAE, RMSE, explained variance, slope/intercept and residual quantiles;
- `action_alignment.csv`: explicitly descriptive action/advantage, action/work and action/realisation associations, intervention rates and sign disagreements;
- `optimizer_updates.csv`: one validated row for every start/complete optimizer event, bound to the exact transition batch and parameter hashes;
- `optimizer_mutations.csv`: paired before/after optimizer mutations with losses, clipping, gradient/update norms and whether the relevant parameter set changed;
- `optimizer_update_summary.csv`: per-batch critic prediction, target, TD/advantage and mutation summaries for detecting a critic that was wrong, clipped or simply did not update;
- `matched_fork_effects.csv`: only preregistered same-snapshot/same-random-stream branch contrasts, with the perturbed head, physical mediator and discounted objective effect;
- `paired_arm_contrasts.csv`: arm differences formed first within `(optimizer seed, world seed)` and only then summarized across optimizer seeds; and
- `trade_credit_latency.csv`: settled-trade effects/correlations by lag on QoL and population, the GAE influence remaining at each lag and independent continuation error;
- `report.json` and `report.md`: provenance, completeness, high-signal findings and denominators.

Mandatory slices are arm, optimizer seed, world seed, capability regime, personality band, population quintile, QoL quintile, turn band and intervention/constraint reason. Population/QoL quintile boundaries and objective reference scales are frozen from the preregistered baseline/evaluation manifest and then applied unchanged to every arm. Every fraction emits numerator and denominator; `ratio_of_sums` and `mean_of_row_ratios` are different named metrics. A zero denominator yields `null` plus `undefined_reason`; calibration slope and explained variance are likewise null for degenerate slices.

The report localises the first broken link in `native demand -> actor cap -> compiler grant -> executed work -> physical receipt -> final service -> QoL/population -> critic target -> final PPO advantage -> parameter update`. Constraint reasons are categorical and may be multi-valued. Observational associations are labelled descriptive. A marginal effect or causal failure is reported only from a matched fork contrast; village decisions are never treated as independent experimental replicates.

Critic reporting has two target classes. The lambda target is reported only as the critic's partly self-bootstrapped optimization target. Independent calibration uses an exactly defined discounted realised prefix in the same raw units, baseline-centred by subtracting `V_t`, and reports any tail separately: `G_prefix(H) = sum_{j=0}^{H-1}(product_{m<j} gamma_m) r_(t+j)`. A terminal prefix has zero tail; a truncated prefix reports the tail estimate and its source but never silently folds the same critic prediction into the independent target. TD/advantage sign is compared with `G_prefix + independently bound tail - V_t`, not with an unsigned raw return.

It flags:

- request-to-grant loss above 10% within a declared compiler-control unit;
- grant-to-realisation loss above 10% within a matching physical good/material unit;
- a critic absolute signed bias above 10% of the objective reference scale;
- a negative matched-fork physical or objective effect while the corresponding final PPO advantage is positive;
- sign disagreement between TD residual/final advantage and the baseline-centred independent return estimate;
- any slice with fewer than 30 decisions or fewer than three optimizer seeds as insufficient.

Single-seed and per-world slices remain useful descriptive diagnostics but can never satisfy the experimental sufficiency rule. Arm effects are the distribution of paired seed/world contrasts, not row-level p-values; the report shows all five optimizer-seed contrasts and does not inflate `n` with villages or turns.

## 6. Data plan

The report manifest assigns every input to exactly one evidence class:

1. **Original training trace**: rows written contemporaneously before a historical optimizer update, with that update's checkpoint, optimizer, normalizer, rollout and update digest. Only this class can explain the actual historical PPO correction.
2. **Checkpoint evaluation**: a bound checkpoint acting on new evaluation worlds. It supports current-policy behavior and independent-return calibration, never historical-update attribution.
3. **Instrumented replay reconstruction**: original snapshot/action/background/RNG histories replayed under a separately hashed instrumented binary. It is admissible only after exact action, observation, reward, state-fingerprint and terminal-event parity with the original binary. Recomputed values/targets are labelled reconstructed.
4. **Prospective diagnostic training trace**: a new run with transition and update tracing enabled. It supports causal claims about that run, not T3's unseen internal history.

Execution order is:

1. Run simulator and learner unit/replay tests and the three benchmark profiles.
2. Run a local diagnostic-only stability sample on at most three non-confirmation seeds. It validates schema, joins, conservation, bounds and report generation; it cannot select an arm, tune a treatment or establish readiness.
3. Preserve the original T3 binary, checkpoints, manifests and result files as immutable evidence and inventory which evidence classes can actually be constructed. The original binary cannot emit the new envelope.
4. On the VM, run the separately named **T3 checkpoint evaluation** and, only if bound histories exist, the **T3 instrumented replay reconstruction** for T3def, T3per and T3lo with the original five paired optimizer seeds per arm. Preserve seed identities across arms and bind both binary hashes plus every checkpoint/normalizer/manifest hash.
5. If original rollout/update traces do not exist, mark “why the historical T3 critic failed to update” unanswerable. A prospective diagnostic rerun is the only substitute and is never relabelled T3. Any readiness-bearing rerun uses the VM, three matched arms and the same five paired optimizer-seed identities; a smaller or differently shaped cohort requires a reviewed pre-registration.
6. Confirmation seeds 3000-3059 remain sealed.

If exact T3 artifacts cannot be bound, the report says so. Evidence classes use separate output directories, titles and provenance; rebuilt or reconstructed evidence is never mixed into a historical causal claim.

## 7. Validation gates for the diagnostic build

- With capture disabled, simulator fingerprints and ordinary bridge payloads are byte-identical to the pre-change contract. With capture enabled, physics/action/reward fingerprints remain identical and only the advertised envelope/schema plus additive sidecar differ.
- Scalar/worker, independent replay and paired fork continuation reproduce every new-transition economic sidecar exactly; restored observations are not duplicated as transitions.
- Conservation checks cover population, food, materials, goods, herds and bilateral trade, including fission, migration, dispatch/reintegration and stock/capital transfer witnesses at the correct pre/post boundary.
- Multi-group herds, zero-demand goods, zero-trade turns, made-to-order partial delivery, storage limits, termination and newly founded villages have focused tests.
- The learner rejects missing, duplicate, mixed-trajectory, future-schema, nonfinite and denominator-inconsistent records. Validation is field-specific: physical counts, stocks, gross flows, durations and declared proportions reject negatives; signed TD residuals, advantages, log-probabilities, net transfers, trade utility and reconciliation residuals permit their documented ranges. Ratios are absent when their denominator is zero, and an intentionally unbounded planner return is encoded as a finite value plus an explicit bound-state flag rather than JSON infinity.
- The aggregator is order-independent and golden-tested on hand-computable fixtures.
- A leakage test perturbs every diagnostic-only field and proves identical actor observations, teacher labels, raw/shaped rewards and sampled actions at fixed RNG state.
- Every test fixture validates the machine-readable field dictionary, missingness reasons, stable good ordering and source-phase/unit tags.

## 8. Proposed two-objective personality learner (not yet implemented)

This section specifies a candidate experiment, not the diagnostic implementation. It removes `welfare` and `trade_utility` from both policy advantage and primary value heads. Welfare and realised trade utility remain mandatory evaluation outcomes. An auxiliary predictor, if later justified, is a disclosed extra treatment and cannot feed policy advantage.

### 8.1 Objective and lineage equations

The new objectives use one centralized lineage transition rather than duplicating a lineage quantity over village rows. A stable `lineageId` is inherited by a daughter at fission; an independently founded institution starts a new lineage. Camps retain their institutional lineage across migration. People are not individually tagged: person-years accrue to the source interval, and migrants participate in the destination camp's lineage from the next decision boundary. Ambiguous ancestry is a trace error.

For lineage `L` and interval `t` of duration `dt`:

- `S0(L,t)` is the set of decision-bearing camps at the opening boundary;
- `S1(L,t)` is the committed closing set, including daughters founded by those decisions;
- `P1_i` is committed closing linked population, counting each resident and source-linked hunter exactly once;
- `PY_i` is authoritative `DemographicReport.PersonYears`, which already accumulates both connected residents and detached source-linked hunters exactly once; and
- `Q1_i` is committed ending QoL.

The raw rewards are:

`R_population(L,t) = sum_i PY_i`

`Qbar(L,t) = sum_i(P1_i * Q1_i) / sum_i P1_i` when closing population is positive, otherwise `0`

`R_qol(L,t) = dt * Qbar(L,t)`

Thus population is an integrated physical count while QoL is invariant to multiplying every headcount by a constant and is not multiplied by the number of settlements. A daughter has no separate opening action on its birth turn but is included in `S1`; on the next turn it joins `S0`. If all descendants go extinct, interval person-years remain credited, `Qbar=0`, the lineage terminates and its bootstrap is zero. Fission, migration and extinction fixtures must hand-calculate these equations. This replaces the current recent-parent grouping and duplicated per-row `qol_years`; the two definitions may not share a name in artifacts.

The actor is factorized by village, but the critic and GAE operate once per lineage transition. All village actions in `S0` receive the same lineage advantage; the PPO surrogate is averaged within lineage-turn and then across lineage-turns so fission cannot increase gradient weight merely by creating more rows. The successor is the next aggregate state of the same lineage, including all surviving daughters. Every person and camp appears in exactly one lineage transition at a boundary.

### 8.2 Policy-conditional critic

The critic has two objective-specific heads on a critic-shared centralized lineage encoder: `V_qol(x_L, p_L)` and `V_population(x_L, p_L)`. These experimental arms use one immutable lineage personality `p_L`, drawn at the root and inherited exactly by daughters; daughter personality drift is disabled in all three arms. The value heads estimate the two raw objectives under that personality-conditioned factorized policy. `p_L` is therefore a critic input and is stored with every rollout; it conditions expected actions but never alters reward definitions or mixes head outputs. Actor code cannot read critic-private lineage features. The constrained extinction/small-settlement cost critic is also conditioned on `(x_L, p_L)`.

On-policy GAE is computed separately in raw units for each head and only then divided by fixed preregistered reference scales. Gamma remains `exp(-dt / 400)` and lambda, reward shaping, guard definition, cost multiplier update, initialization, data order and stopping/selection rules are frozen across arms. Reference scales and normalizer moments come from a named T3-independent calibration artifact and never from the current minibatch. Raw, shaped and scaled quantities remain separately traceable.

### 8.3 Personality attention

The two-component personality is a validated simplex `p=(p_qol,p_population)`, with non-negative finite values summing to one. For comparison with the existing four-component personality `(p_welfare,p_population,p_trade,p_qol)`, the preregistered projection is `p_qol' = p_qol + p_trade + 0.5*p_welfare` and `p_population' = p_population + 0.5*p_welfare`. This is a deliberate experimental projection, not a claim that welfare or trade utility decomposes uniquely; A-to-B estimates the entire objective/projection package.

The actor state encoder produces `h`. Two non-exchangeable objective slots use separately named projections and fixed orthogonal identity embeddings: `z_qol = W_qol h + e_qol` and `z_population = W_population h + e_population`. Attention scores are `content_k(h,z_k) + log(p_k)`. An exact zero personality component masks that slot to negative infinity; `[1,0]` and `[0,1]` therefore select pure QoL and pure population contexts, while `[0,0]` is invalid. The policy representation is `LayerNorm(h + W_context * sum_k alpha_k z_k)`. Personality has no other actor path, and every active strategic, service, route, movement, fission, commitment and trade-posture head consumes this representation. Capability masks remain external legality masks.

Tests cover exact endpoints, monotonic attention-weight ratios over a fixed personality grid, slot identity under checkpoint save/load, gradients reaching the named slot associated with each pure endpoint, and the absence of a direct personality/diagnostic bypass. Learned action monotonicity is reported, not assumed.

### 8.4 Credit and experimental design

The final policy advantage is `p_qol*A_qol/s_qol + p_population*A_population/s_population`, followed by the same guard-cost subtraction and regime balancing in every arm. There is no product-form welfare advantage and no direct trade-utility advantage. Trade can receive immediate QoL/population credit when settlement precedes needs and delayed credit thereafter. The diagnostic report must measure trade-to-objective latency, realised-prefix/tail error and remaining `gamma*lambda` influence before removal is authorized; an auxiliary trade loss is not part of the initial experiment.

All three Round 4 arms use the same centralized transition topology, critic conditioning, data, compute, guard and checkpoint-selection rule:

- A: four-objective basis and existing FiLM;
- B: two-objective basis and two-component FiLM; and
- C: two-objective basis and the objective-attention replacement above.

A-to-B estimates the preregistered objective/remapping package, not the isolated effect of deleting a value head; B-to-C estimates attention conditional on that package. Three arms do not estimate the full interaction. Isolating direct policy credit from auxiliary value prediction requires an additional arm and review.

This is a new lineage, not a resume-compatible checkpoint edit. Encoder migration requires an explicit parameter manifest; objective heads, personality conditioning and affected optimizer state are reinitialized identically across arms. Training requires a reviewed pre-registration and the VM cohort of three matched arms by five paired optimizer seeds with identities preserved. Confirmation seeds 3000-3059 remain sealed.

## 9. Review questions

The architecture reviewer must answer:

1. Are the four economic ledgers sufficient to localise actor, compiler, resolver and critic failures without leaking privileged state into the actor?
2. Which ratios or denominators are misleading and what additional conservation witnesses are required?
3. Can QoL/population alone give trade a usable credit path at the current horizon and discount, or is an auxiliary (non-advantage) trade prediction loss needed?
4. Should objective attention replace FiLM, supplement it, or be tested as a separate arm?
5. What minimal three-arm Round 4 design identifies the value-basis and conditioning effects without sampling bias?
