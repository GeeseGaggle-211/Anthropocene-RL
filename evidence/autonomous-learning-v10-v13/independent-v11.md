# Independent Astra/high review: V11 objective calibration

## Verdict and exact scope

**GO for calibration only**, using `configs/training/autonomous-reward-calibration-v11-plan.json`, SHA256 `28f3dc563a4fa43feb1f915231bbafb82c25a4532a94f45c94e0108942a6340c`. All 227 input hashes independently matched their files. The native and learner reward-contract documents are byte-identical. The superseded draft is not the cleared configuration.

This clears shared physical collection of at most 2250 native resolves plus 2250 replay resolves and nine objective-specific live-critic fits, each at most 128 epochs with checks every 32. It does **not** clear retained actor/PPO updates, automatic continuation, economic promotion, or broad scale-out. Historical work remains 51913 resolves and 1695.8695237250067 active seconds. The separate V11 envelope permits an additional 24000 resolves and 1200 active seconds: cumulative ceilings 75913 and 2895.8695237250067. Attempted failed work remains charged.

Review was source, configuration, hash, and existing-test inspection only. The reviewer executed no native environment, model construction/forward/sampling, fitting, optimizer update, or new tests. The eight focused tests are implementation-provided evidence, inspected rather than independently rerun.

## Mechanics reviewed

The three arms are continued original reward, midpoint-population interval reward, and fixed alpha=1.5 population reward with reference population 100. Native physics, actor observation width 1062, action4 authority, focal native evaluation welfare, H400 discounting, and the current actor architecture remain unchanged. Model/checkpoint8 versions the learner objective and value context; it imports every model7 parameter and existing optimizer state. Each arm begins from the same corresponding seed actor, actor Adam, sampler/shuffle state and counters. The three actors are independently initialized historically, but seed307 retains the already-disclosed shared critic ancestry from V9; V11 does not repair that statistical dependence.

The shared physical calibration uses the same 23 fitting starts in worlds886001–886004 and six family-balanced validation starts in world886006 per actor, with newly fixed sampler seeds. The collector sorts completed worker jobs into original order before flattening rows. The arm loop deep-copies the physical episode and its corresponding row slice, reconstructs that arm's rewards and immutable initial-population context, and revalues every row and final nonterminal observation before constructing its frozen targets. No fitting targets are taken from world886006 or DEV worlds886007/886008. Early extinction is retained, not replaced with successful trajectories.

The reference-population factorization is algebraically valid: r(P0)=c*r(Pref), c=(Pref/P0)^(alpha−1), so the same factor applies to rewards, values, next values, tails and advantages before ordinary PPO processing. Model8 applies c to the full model7 residual value expression, including fixed offset/scale. The factor-aware intercept uses sum(a*residual)/sum(a²), a=value_scale*c, which is the correct one-parameter TRAIN least-squares adjustment. Critic loss remains in actual objective units divided by the fixed inherited value scale squared. No validation-derived rescaling is introduced.

P0 is stored as explicit own reset metadata rather than inferred from compressed neural features. Current counts come from completed owner history receipts. Context construction preserves P0 separately from current population; model8 rejects missing, nonfinite or malformed context, nonpositive P0/factors, and negative current population/potential. Reset transport checks the exact-integer range of the native float population coordinate. The model8 actor still consumes its original observation tensors; the added context affects value accounting only.

The bounded potential implementation uses Phi=min(P,100), terminal absorbing Phi=0, and gamma*Phi(next)−Phi(now), with V_shaped=V_base−Phi. The inspected terminal/truncation tests establish telescoping and equal GAE under physical-time discounting. It is correctly treated as an equivalence test rather than another expensive expected-improvement arm. Optional Crowding headcount extraction excludes other fission triggers, emigrants, starvation deaths and daughter welfare; no Crowding reward is trained in this clearance.

Calibration protects actor/heads, actor optimizer state, frozen critic anchors, value buffers, sampler/shuffle state, counters and configuration. The live critic and its optimizer may change. All nine must meet the original finite strict criterion: EV>=0.60, MSE/Var(return)<=0.40, absolute raw-unit bias<=2, and MSE no worse than their own entry. Degenerate validation variance fails. TRAIN intercept corrections are bounded to the stated before/after applications; no until-pass extension is cleared.

## Required interpretation and subsequent gate

1. World886006 is world-disjoint from fitting but has been used in prior V10 calibration. Its new sampling is not a new independent holdout. Validation is also used for the predeclared calibration stopping rule. Final reused DEV is a matched diagnostic, not untouched promotion evidence.
2. The c factor is an exact reward/return identity, but an observation-only reference critic need not be conditionally sufficient in a partially observed environment: P0 can retain information missing from current x. The contract correctly discloses this restriction. Report raw errors and P0-stratified coverage; do not claim the factor proves absence of aliasing. A critic-only context input remains a possible later repair if actual residual evidence warrants it.
3. Objective-specific tails are revalued consistently but start from an inherited critic trained under the old objective. For midpoint/superlinear they are approximations, not ground-truth new-objective infinite returns. Preserve and inspect zero/doubled-tail caches and censoring before interpreting gate success. A high bootstrap-dependent EV alone is insufficient for economic scale-out.
4. Midpoint reward is a concrete interval proxy, not an exact final-horizon population objective. Alpha=1.5 can favor population at the expense of QoL. Judge matched endpoints using unchanged native focal population, last-realized and extinction-aware QoL, welfare, extinction, trade and resident outflow accounting. The user seeks joint population/QoL improvement, not heuristic parity.
5. Retained training needs a separately pinned configuration and review of actual nine-critic results. The inspected future fixed-row collector fills early-terminal shortfalls before each update, preserving the intended 224/224/224/224/128 row boundaries for 1024 transitions per cell. Actual active actor minibatches can still differ due to KL stopping and must be reported rather than equated with scheduled budgets.
6. Model8 inference/save-resume integration requires objective-episode context alongside native state when eventually claiming a deployable model8 plug. The current calibration clearance establishes checkpoint and objective context mechanics, not a new native model8 plug proof.

The V10 positive learning and signed-reversal evidence continues to argue against declaring the MLP frozen or globally saturated. V10 joint endpoint QoL and narrow seed307 current-value failures remain historical failures. V11 calibration is a principled bounded objective experiment, not evidence that reward mismatch is already the sole cause or that a GRU/architecture expansion is necessary.

## Prospective entry-ready restoration advisory

During calibration, root reports that some critics already passed every absolute readiness threshold at entry, then their required fitting worsened validation MSE relative to entry. Those fitted results correctly fail the unchanged nonworse-entry condition; their failures must remain visible.

A separately pinned, cached-only repair may restore the exact ENTRY state for every arm whose entry already passed the unchanged strict criterion. Apply this deterministic rule uniformly rather than selecting a favorable historical fitted epoch. Reconstruct entry from the pinned model7 parent and exact objective mapping, preserving original model/optimizer/RNG/counters, and verify raw-value parity against the saved entry cache. Keep the original frozen objective rewards, targets and bootstrap tails. Do not regenerate targets after restoration. Record restored-entry readiness separately from fitted readiness; the validation is reused and this is not independent confirmation.

Finish assessing all nine original results before proposing repairs for entry-failing arms. Further intercept or finite fitting attempts require their own prospectively fixed TRAIN-only rule and unchanged gate; no automatic until-pass iteration or retrospective best-epoch choice is cleared. This advisory clears no execution and changes no original failed result or resource ledger.

## Completed calibration and concrete restoration preflight

The original run completed with `REVIEW_REQUIRED_CRITIC`: 2143 physical collection resolves plus 2143 replay resolves and 137.72793102399737 active seconds, cumulative 56199 resolves and 1833.59745513901 seconds. Only the fitted seed101 and seed307 superlinear attempts passed; the other seven fitted attempts failed the unchanged gate. These results remain authoritative for those fitted checkpoints.

Independent cached arithmetic reconstructed all 6429 arm-specific rows from owner receipts, immutable P0, and the declared objective equations. Maximum reward reconstruction discrepancy was 7.105427357601002e-15. All n-step, zero-tail, doubled-tail and tail-weight arrays reconstructed exactly. Row population context and reference factors matched their receipts, and the native focal reward identity remained valid. Entry metrics independently reproduced to reported rounding without model forwards:

| Seed | Objective | Entry EV | Entry MSE/variance | Entry bias |
|---|---|---:|---:|---:|
|101|Original|.825181|.178913|+.655122|
|101|Midpoint|.790335|.221331|+1.026802|
|101|Superlinear|.831342|.174700|+.758013|
|211|Original|.849782|.150706|−.240960|
|211|Midpoint|.829643|.170412|+.076951|
|211|Superlinear|.820430|.179627|−.083416|
|307|Original|.916931|.093193|+1.434884|
|307|Midpoint|.910960|.107622|+1.842795|
|307|Superlinear|.766644|.244483|+1.328781|

**Restoration-only GO** for `configs/training/autonomous-objective-entry-restoration-v11-plan.json`, SHA256 `244abd3bccf180fcfcc3d4c87a07fb92eb489f798c2e33a2545e083fda73040a`. All 247 pins independently matched, including all nine objective caches and fitted-attempt checkpoints. `scripts/restore_ready_objective_entries_v11.py` implements the uniform rule: all nine entries must already pass; restore every entry rather than choose between fitted and original values. It compares the complete model and populated optimizer to the original parent, checks protected actor/RNG/counters, requires bit-exact per-episode cached values, and reapplies the same strict gate including nonworse MSE without epsilon. No target/tail refresh or fit occurs. Provenance binds the unchanged target cache and new restoration configuration.

Cleared scope is zero native resolves, zero fitting/optimizer/PPO updates, and at most 60 active seconds within the existing V11 envelope. Restoration is not permission for the separate retained-training stage. It does not erase the seven fitted failures, the previous seed307 world886005 current-value failure, or the validation reuse/partial-observation/bootstrap limitations above. Any actual restored READY result is scoped to this reused six-family world886006 gate.

## Actual entry restoration and first1024 comparison clearance

Restoration completed READY for all nine entries with zero native resolves and zero fitting, using 6.61580105900066 active seconds. Cumulative work is 56199 resolves and 1840.2132565680222 seconds. The reviewer independently loaded the nine saved checkpoints and original model7 parents and compared complete model tensors, populated optimizer states, sampler/shuffle RNG and episode/transition/update counters: all exact. Recorded per-episode cached-prediction differences are zero.

Aggregate readiness is not uniformly good conditional readiness. In the low-P0 stratum, seed101 superlinear has EV .029609 and MSE/variance 1.458440; seed307 superlinear has bias +3.206243 and MSE/variance 1.269950. Original and midpoint also miss some conditional thresholds, including seed307 original bias +2.347419 and midpoint +2.467540. These are real limitations, not proof that P0 factorization alone caused the error or that new critic inputs are already necessary. Preserve them when interpreting subsequent improvements or regressions.

**GO for the limited first1024 comparison** under `configs/training/autonomous-reward-first1024-v11-plan.json`, SHA256 `1214808de17f0aaaa6325bdc29b84bc609bb6720667c072ccbab3ecb49a1db33`. All 242 pins independently matched. Scope is three arms times three seeds times exactly1024 retained transitions (9216 training resolves), plus at most2700 final DEV and2700 replay resolves. The projected maximum cumulative resolves are70815, below75913; remaining shared active-time allowance is1055.656267 seconds. No automatic4096 continuation or broad promotion is cleared.

The fixed-row loop fills short episodes before updating, yielding four224-row updates and one128-row update per cell. This equalizes retained rows and scheduled updates; realized episode starts can diverge across arms when learned actions change termination, and actual active minibatches can differ through KL stopping. Report these distinctions. Checkpoints precede optional diagnostics, all actual objective-context rollout rows are exported, and the existing64+64 signed probes retain their fixed criterion and dose.

The configuration's `baseline_panel` is the historical pre-V10 panel stored in the first1024 V10 summary. It is not the immediate starting policy for V11. For V11 incremental endpoint comparisons, use the already-pinned `parent_gate` final4096 V10 evaluations as the immediate parent baseline, and compare each reward variant to the newly trained original-control arm at the same additional budget. Label the earlier baseline historical. All panels reuse the same12 DEV starts and declared sampler convention; none is a new holdout.

Existing cached potential-equivalence evidence covers all12288 V10 rows, reports maximum advantage discrepancy6.0396e-14 and corrected-return discrepancy6.3949e-14, and records zero native/model/optimizer calls. It supports treating exact potential shaping as the mathematical equivalence arm; it supplies no separate economic improvement evidence.

This is authorization for a bounded diagnostic learning comparison despite documented conditional critic inaccuracies, not a claim of broad critic readiness. Actual endpoint population/QoL, extinction, training stability and current-rollout value health must determine the next verdict. Seed307's known world886005 failure remains unresolved.

## Actual first1024 comparison: independent cached assessment

The stage completed all nine cells and correctly stopped at `STAGED_REVIEW_REQUIRED`. Work was9216 retained training resolves plus2667 DEV and2667 replay resolves,539.2975817960105 active seconds. Cumulative work is70749 resolves and2379.5108385850326 seconds, leaving5164 resolves and516.358685140 seconds in the unchanged V11 envelope.

The reviewer independently checked all45 audit-file hashes and nine final checkpoint hashes, objective/checkpoint counters, all9216 finite real rollout rows, and exact224/224/224/224/128 row boundaries per cell. Independently reconstructed physical-time GAE and returns matched exactly; post-update MSE reconstructed from saved predictions/targets matched exactly. Actual actor minibatches were1152/1152 active. In-sample post-update MSE/target-variance ranged .02719395–.13634440. Every action head had a positive stage gradient maximum. All nine fixed64-forward/64-reverse inherited-Adam probes had positive signed final gaps in both directions. These are genuine responsive learning mechanics, not evidence of global freezing, numerical saturation, or an architecture bottleneck. They do not by themselves establish out-of-sample value readiness.

Native endpoint and annual-welfare calculations independently reproduce the reports:

| Arm | Mean endpoint population | Extinction-aware endpoint QoL | Mean native annual welfare | Extinctions /36 | Trades |
|---|---:|---:|---:|---:|---:|
|Continued original|1.833333|.130980|7.339917|8|3|
|Midpoint|1.722222|.120079|6.755953|9|2|
|Alpha1.5|2.083333|.226155|7.543082|6|3|

The alpha arm is promising but does not dominate uniformly. Against the matched continued-original arm, seed101 improves endpoint population/QoL and reduces extinction, but native annual welfare falls slightly8.136097→8.091182. Seed211 improves mean endpoint population/QoL and annual welfare, but extinctions increase3→5. Seed307 improves QoL and reduces extinction4→1 while endpoint population falls1.583333→1.5 and annual welfare6.932094→6.857349. Aggregate means must not conceal these tradeoffs. Midpoint supplies no persuasive aggregate advantage at this dose. There is no evidence supporting a GRU or actor expansion from these results.

## Recommended next gate, not execution clearance

The proposed new-world no-update comparison is the appropriate minimal next experiment: two genuinely new worlds, six physical families each, continued-original versus alpha1.5 for all three retained actors, and the explicit model8 two-Elder native/save/RNG-sidecar plug. At most300 generation+300 replay,1800 evaluation+1800 replay, and8 plug+8 replay resolves total4216, within5164 remaining. Concrete source/configuration/pins and actual physical admission still require prospective review before launch.

Predeclare world/age selection and physical-gap handling without model or welfare selection. Use the same scenario and sampling schedule across control/alpha. Reuse these evaluation trajectories to save current objective-specific n-step, zero-tail and doubled-tail targets and predictions; no extra native collection is necessary. Do not fit or select checkpoints on these new worlds. Preserve full reward/P0/value context and report seed/family/P0 strata, endpoint population and extinction-aware QoL, native welfare, extinctions, and resident outflows. Specify endpoint/critic decision criteria prospectively, rather than devising a passing rule after outcomes arrive. The prior aggregate strict critic criterion remains distinct from conditional failures and tail sensitivity.

A successful fresh comparison could justify a bounded subsequent training-scale increase under the tested native/action/architecture contract. Two worlds and sparse trade events cannot establish broad ecological/domain competence. A failed or mixed comparison should identify whether objective tradeoffs, conditional value errors, exploration or insufficient learning are implicated; it should not default to an architecture-infeasibility diagnosis.

Source note for the new inference adapter: `restore_sidecar` should apply the same canonical village-identity validation as `start_episode`, before committing RNG/state. Its direct int-key dictionary conversion currently silently collapses aliases such as'0'/'00' and accepts negative IDs. This is a concrete corruption-contract hardening request for the future plug freeze, independent of already-completed training.

## Model8 native plug preflight and prospective fresh criterion

**Plug-only GO** for `configs/training/autonomous-model8-plug-v11-plan.json`, SHA256 `9649dccb4c1297b6c2985318e904d9f57f6fa29fa7da744c03314a2a97d7c75d`. All247 pins matched. Scope is the exact retained seed101/211 alpha checkpoints, fitting TRAIN trade pair886001/d2, eight native plus eight replay resolves, no PPO, and at most60 active seconds within the same V11 ledger.

The inference adapter now validates sidecar village identities consistently with episode initialization, rejecting negative IDs and alias collisions before committing restored state/RNG. Inspected tests explicitly preserve P0 and sampler state. The plug driver uses separate owner-only controllers and RNG streams, performs two four-boundary runs, and at the second run's midpoint captures native state, writes learner objective/P0/RNG sidecars to disk, constructs fresh controllers from their checkpoints, and restores those sidecars after native reset. It compares actual actions, rewards, completed histories, staffing, observations and fingerprints exactly across both full runs, verifies local welfare identities and replay, and checks retained checkpoint hashes unchanged. Since P0 affects the critic but not actor actions, action equality by itself is not a P0 preservation proof; the explicit sidecar state semantics and tests provide that separate evidence.

The proposed fresh two-world decision criterion is defensible for a bounded next training expansion: alpha pooled mean endpoint population, extinction-aware endpoint QoL and original integrated native welfare each at least the matched continued-original control; extinction count no greater; all three alpha critics meet EV>=.60, MSE/return-variance<=.40 and absolute raw bias<=2 on current fresh trajectories; existing fixed reversal, finite-state and actual plug checks pass. This should be frozen before new outcomes. It is a point-estimate gate over two worlds, not a statistical superiority claim or broad-domain promotion. Per-seed extinction regressions, family/P0 residuals and zero/doubled-tail sensitivity remain reportable limitations even if pooled criteria pass.

Integrated native welfare is the appropriate primary welfare comparison. Annual welfare divided by only actually realized years can obscure early extinction; fixed500-year annualization with zero contribution after extinction is a separate clearly labeled quantity. Root reports existing36-episode mean integrated welfare3619.8068 control versus3725.5183 alpha; this is consistent with the modest aggregate advantage and does not remove seed-level tradeoffs.

## Actual model8 plug and fresh-generation-only clearance

The model8 plug completed PASS with8 native+8 replay resolves and2.231250145996455 active seconds; cumulative70765/2381.7420891210204. Independently read both four-boundary trace arrays: they are exactly equal, and the trace digest matches `ecd8b2ca4a7a4f17a68e27fc0848b02cd1c094da7d79efd1e44152bb8cf11589`. The on-disk sidecar digest matches `ab7c6f380449c0bbd644f97a1b4c82ef74aca0cf59f93b1d5b37fcee1871cf6d`. Separate controllers retain checkpoint/objective identity, immutable initial populations276 and404, and their5056-byte sampler states. This supports actual owner-isolated model8 WEGO/save/resume/replay plug readiness; it adds no economic training evidence.

**Generation-only GO** for `configs/curriculum/autonomous-fresh-validation-v11-envelope.json`, SHA256 `50767ade3b10b968b5b94f41c6d870c2d3e087129133d697b9ebf0748cbdfd9a`. All345 input pins match, including the new isolated generator DLL/project/source and simulation-source files. The fresh-validation contract mirrors are byte-identical. A direct diff against the preserved `AutonomousOpportunityCurriculum.cs` confirms changes only to class name,12-versus48 trajectories/two-versus-eight worlds,300-versus1200 resolve limits,120-versus600 seconds, and associated guard messages. Scripted policy and simulation mechanics are unchanged.

The pinned generation plan contains exactly six families for each new validation world886009/886010 and snapshot ages0/2/8/25. The wrapper reserves the full300 generation+300 replay allowance before launching, uses the lesser of120 seconds and the remaining shared deadline, verifies manifest/replay counts and snapshot digests, and preserves subprocess failure/timeout output. No generation attempt is erased or refunded. This clearance excludes admission, model evaluation and any parameter update; those next stages require their concrete artifacts and preflight. Predeclared age/fallback slots and complete-family admission remain binding without welfare-based replacement.

## Fresh generation actuals and admission-only preflight

Generation completed PASS:48 snapshots, twelve trajectories each25 decisions,300 generation+300 replay resolves, no unavailable rows or failures,17.059131257992703 active seconds. Cumulative71365 resolves/2398.8012206090207 seconds. The reviewer independently verified raw manifest SHA256 `1c89b526d8daa9ad95e8389279688def8e827a2d32d48d8b866162459dfdcde1`, every snapshot digest, all twelve replay-PASS trajectory records, and validation-only partition membership in worlds886009/886010.

**Admission-only GO** for `configs/curriculum/autonomous-fresh-validation-v11-admission.json`, SHA256 `587297abf6aab9fc5bd77579641078f8829a8b901c201d80029465f11f3943f3`. All345 pins match; embedded twelve slots exactly equal the preregistered slot file. The admit branch creates no neural model or optimizer. It uses schema/reset and owner-address reads under an empty native-step category budget, with at most60 active seconds inside the shared deadline. It enforces fresh world/validation partition, full six-family coverage per world, and pinned ordered ages. Eligibility/fallback uses physical masks, positive owned animals for husbandry, fishing capability, and mutually known reciprocal goods stocks for trade. No outcome, reward or neural score chooses starts. This clearance includes no evaluation or fitting.

## Actual admission and final fresh-evaluation-only clearance

Admission passed all twelve first-choice ages exactly, with no fallback, no neural calls and no native steps, adding1.3587863099965034 active seconds. Cumulative71365 resolves/2400.1600074690214 seconds. The reviewer checked the selected identities against each frozen preferred slot, including reciprocal trade opportunities and actual husbandry/fishing masks.

**Evaluation-only GO** for `configs/training/autonomous-fresh-validation-current-value-v11-plan.json`, SHA256 `37d955e590d8f0560a55f1743920acfb1b35bd9735fba3d9a54605405e9a72b7`. All352 pins matched. Scope is the six exact retained original/alpha model8 checkpoints, all twelve admitted starts per policy,25 decisions, common seed+510000 sampling convention, at most1800 native+1800 replay resolves and360 active seconds. Projected cumulative resolves74965 remain below75913, and the stage deadline is within the remaining shared allowance. No fitting, PPO, new checkpoint selection or automatic expansion is cleared.

The evaluated source verifies parent COMPLETE status,1024 durable additional transitions, checkpoint digest and exact counters. It saves actual episode rows, objective context and current predictions with objective-specific n-step/zero/doubled tails from those same trajectories, checks retained state hashes unchanged, and reports original integrated welfare independently of learner reward. Every observed interval must be20 years before reporting fixed500-year annualization. It stops at REVIEW_REQUIRED; the frozen endpoint/critic criteria and conditional/tail evidence must be assessed after actual results.

## Actual fresh evaluation: endpoint pass, critic NO-GO

Fresh evaluation completed1792 native+1792 replay resolves with zero retained updates,58.06848739400448 active seconds. Cumulative74949 resolves/2458.228495364019 seconds;964 resolves and437.641028 seconds remain in V11. Independent cached review verified all six cache and retained-checkpoint digests,1792 finite rows, exact n-step/zero/doubled-tail/tail-weight reconstruction, and reported metrics. The reviewer also independently rebuilt all six protected hashes directly from saved model/optimizer/RNG/counter/config tensors, matching the evaluation hashes exactly without model construction or forwards.

The predeclared pooled endpoint criterion passes: alpha versus continued original gives endpoint population2.416667 versus1.666667, extinction-aware QoL .246566 versus.188254, mean integrated original welfare5023.234988 versus4146.629857, and3 versus3 extinctions. Alpha population and integrated welfare improve for all three seeds. Seed101 QoL nevertheless falls .241004→.148462; seed307 extinction rises0→1 while seed211 falls2→1. These differences remain material qualifications to pooled success.

The overall predeclared candidate gate is **NO-GO_CRITIC**, not a bounded-expansion GO yet:

| Alpha seed | Fresh EV | MSE/variance | Raw bias | Strict result |
|---|---:|---:|---:|---|
|101|.330112|.685776|−2.651116|FAIL|
|211|.544575|.462163|−1.251367|FAIL|
|307|.765996|.235533|−.409384|PASS|

The failures persist under both zero-tail and doubled-tail targets, so changing bootstrap conventions cannot honestly repair them. Seed101 zero/double EV is.318162/.330886; seed211 .539487/.537982. This is not evidence of a purely tail-driven threshold miss.

A single high-P0 context, `validation-886010-trade-d0` with P0=340, contributes97.39% of seed101 squared error and94.61% of seed211 squared error. Its opening prediction/target pairs are29.125/213.631 and41.982/141.449. Seed101 makes no settled trade in that trajectory despite high realized welfare, so this must be called a trade-family physical context, not automatically a failed trade-credit mechanism. All three alpha low-P0 aggregate strata now pass the absolute thresholds; the fresh failures do not support claiming the external P0 factor is specifically breaking low-population prediction. Coverage, stochastic outcome variation and conditional value generalization remain plausible explanations. There is no numerical-saturation or GRU-necessity evidence here, and a single sampled return is not proof of the conditional expected value.

## Recommended finite critic-only repair, no execution clearance

Preserve the already-passing seed307 unchanged. A more discriminating use of the same900-resolve allowance is nine current-policy TRAIN episodes for each failed seed101/211,25 decisions and replay (450+450 maximum): six family starts plus three additional distinct trade-family starts across the four existing fitting worlds. Choose all starts prospectively from physical/P0/age coverage, not by searching returns. This preserves family coverage while addressing the dominant observed context gap better than spending a third of the allowance unnecessarily refitting307.

Use one prospectively fixed TRAIN-only fitting dose, such as64 epochs, with no retrospective best-epoch selection or until-pass extension. Preserve actors/heads, actor Adam, fixed anchors/value buffers, RNG and counters exactly. Freeze the collected objective targets/tails before fitting. Evaluate only against the already-saved fresh validation targets; never regenerate validation tails, fit validation labels, alter denominator/thresholds, or drop the difficult trajectory. Preserve all original fresh failures. A passing repaired result would be reused-validation calibration evidence; the fresh panel has now informed the repair and cannot be called independent confirmation. Concrete source/configuration/pins are required before this repair is cleared. Any further native collection beyond the remaining budget needs a separate prospective envelope with the existing ledger intact.

## Concrete fixed critic-only repair preflight

**One-attempt repair GO** for `configs/training/autonomous-fixed-current-critic-repair-v11-plan.json`, SHA256 `f506d727fd8e9eaab9bbd5e38073ddada00a4a4e4c568d807f25f92ef7285abb`. All359 pins match and native/learner repair contracts are byte-identical. The nine immutable fitting starts cover six families in886001 plus distinct trade-d0 starts in886002/003/004; the additional three trade starts each have P0=340, with both own trade roles represented. All are TRAIN in existing fitting worlds. No validation world9/10 labels or actions enter fitting.

Only failed critics101/211 are restored and adapted; passing307 is pinned and unchanged. Source `fixed_fit` has no validation argument and performs exactly64 epochs with minibatches32, inherited critic Adam/LR, gradient clip.5, and separate fixed TRAIN factor-aware intercepts before and after. It cannot stop early on validation or choose a best epoch. The actor/heads, actor Adam, frozen anchors/value buffers, sampler/shuffle RNG and counters are protected. New fitting targets use each entry critic's frozen tails; existing fresh validation target/tail arrays remain unchanged. Saved final predictions permit independent raw metric, family/P0 and zero/doubled-tail analysis. Strict absolute and nonworse-entry conditions remain unchanged.

Scope is at most450 native fitting-data resolves+450 replay and120 active seconds inside the unchanged envelope, projected75849<=75913. No actor/PPO update, automatic retry, additional target refresh, or expansion is cleared. A pass must be labeled reused-validation frozen-target repair, preserving the original independent-panel failures and not claiming independent post-repair validation.

## Final fixed repair actuals and V11 verdict

The single fixed repair completed both64-epoch doses and returned **NO-GO_CRITIC**. It used443 native+443 replay resolves and18.758816785993986 active seconds, reaching75835 resolves/2476.987312771016 seconds. V11 has78 resolves/418.882211 seconds left; these do not authorize a new substantial native experiment or reset any previous allowance.

Independent cached inspection verified all fitting/prediction/checkpoint hashes,224 seed101 and219 seed211 fitting rows, exact reconstruction of fitting targets from fixed entry tails, unchanged validation target arrays, all64 epoch records, and final raw metrics. Complete non-live-critic model tensors, actor optimizer group and populated actor Adam, sampler/shuffle, counters and configuration match the original retained checkpoints exactly. Seed307 checkpoint bytes remain unchanged. The original actors and therefore measured endpoint outcomes are unchanged by this repair.

| Failed critic | Entry EV→final | Entry MSE/variance→final | Final bias | TRAIN MSE entry→final |
|---|---:|---:|---:|---:|
|101|.330112→.460883|.685776→.546972|−1.864099|20.065735→.218476|
|211|.544575→.521953|.462163→.488257|−1.540323|27.578140→.062008|

Both remain below the unchanged absolute readiness criterion; seed211 also worsens versus entry. Zero/doubled-tail variants still fail:101 MSE/variance .544923/.568987,211 .478588/.519156. No tail convention or reused-validation relabeling rescues the gate.

The same high-P0 trade-family context contributes96.34% and89.48% of final squared error. A decisive coverage limitation is visible in the actual fitting returns: the largest target anywhere in seed101's nine new fitting episodes is73.8201 and seed211's is71.5086, while the troublesome fresh opening targets are213.631 and141.449. The attempted repair therefore did not furnish comparable high-return examples. Seed211 also worsened its low-P0 stratum to MSE/variance .560792, so aggregate fitting success did not uniformly preserve conditional generalization.

**Final scoped verdict:** native simulator/action authority, owner isolation, model8 plug/save/RNG/replay, responsive finite PPO, and fixed-dose signed workforce reversibility are GO on tested mechanics. Alpha1.5 passes the preregistered pooled fresh endpoint point-estimate criterion, with disclosed seed-level QoL/extinction tradeoffs. **Scaled-training promotion under the declared combined gate is NO-GO because two current critics remain unready.** Broad economic competence is not established. This is not an architecture-infeasibility, actor saturation, global freezing, or GRU-required verdict.

Tiny TRAIN errors demonstrate that the critic can fit these observed samples. They do not establish its ability to generalize to the missing return range, distinguish systematic conditional bias from one unusually successful stochastic path, or prove that the owner observation aliases relevant state. Existing single-rollout data cannot identify the conditional expected return or its sampling uncertainty. The dominant fresh failure is now a large-P0 context; it does not establish a need to append P0 or enlarge the actor.

## Recommended next bounded gate

A new prospective coverage-and-stochastic-uncertainty envelope is defensible; continued fitting epochs on the same443 rows is not the next useful intervention. Preserve the V11 failure and ledger, retain current actors, use no fresh9/10 labels for fitting, and do not change physics, objective, normalizer, architecture or readiness thresholds merely to obtain a pass.

A concrete first diagnostic design is four existing TRAIN trade resets across fitting worlds times four independently fixed sampler seeds times25 decisions for the two failed actors: at most800 native+800 replay resolves. Keep all draws, including low welfare and early deaths, and record identical reset observations and immutable P0. This measures within-state policy/environment-path return variation separately from between-world variation. Because repeats on old TRAIN alone cannot tell whether the specific fresh target was an outlier, a separately declared diagnostic-only replication of the already-consumed fresh trade reset—eight sampler draws per failed actor, at most400 native+400 replay—would directly test that question without fitting those labels. Combined diagnostic maximum2400 resolves; root must freeze actual source/data/resource/time pins before any launch. This is an example bounded design, not execution clearance or a requirement to spend the full amount.

Report per-reset mean, dispersion and uncertainty of the frozen-tail returns, plus zero/double-tail variants. Eight draws are a modest diagnostic, not a reliable estimate of rare-event tails. If comparable high returns occur repeatedly in TRAIN, expand current-policy fitting with those complete trajectories and predeclare world-balanced fitting/regularization choices using TRAIN-only world-disjoint cross-validation. If the TRAIN distribution still lacks the relevant range, generate additional genuinely new TRAIN worlds instead of repeatedly oversampling an unrepresentative bank. Do not select or discard trajectories by observed returns.

If fresh replication shows a high conditional mean, the failure is systematic value generalization and warrants more representative TRAIN coverage before architecture work. If it shows that the single target was unusually high, that supports an uncertainty diagnosis but does not retroactively pass V11's frozen single-path gate. Any new expected-value/replicated-target criterion must be explicitly preregistered and independently evaluated, not quietly substituted for the failed criterion.

A successful later fit needs a separately untouched confirmation cohort, since fresh9/10 already informed the repair. Retain and report the old frozen-target gate as regression evidence. Only if representative TRAIN coverage, stochastic uncertainty, and controlled generalization tests continue to show residual error associated with omitted owner-observable information should a critic-only representation change be proposed. GRU/actor expansion requires separate evidence that relevant history is missing from the current receipt inputs; none of the present results establishes that.

### Prospective broader coverage design advisory

Root proposes a new explicitly chained18,000-resolve/900-active-second envelope, retaining V11's75835/2476.987313 history. A mixed replication design is reasonable: four replicas of nine old TRAIN slots plus two replicas of24 starts from four genuinely new TRAIN worlds gives84 episodes per failed actor,4200 native+4200 replay. Six-world source generation adds900+900. Confirmation of three alpha actors on twelve starts, with four draws for each of two trade starts and one draw for each of the other ten starts, gives54 episodes total,1350+1350. Combined maximum12900 resolves, plus an optional16-resolve plug if prospectively justified. This advisory is not execution clearance.

Keep the primary strict critic gate on the first fixed draw of each of the twelve confirmation starts, preserving balanced six-family/two-world weighting. Treat additional trade repeats as diagnostic/secondary and do not pool them into the primary variance denominator or weights. Two/four repeats provide noisy descriptive variability, not a reliable noise ceiling. If only alpha policies are evaluated on confirmation worlds, that panel confirms value accuracy, not matched endpoint superiority over an unevaluated original control.

Restore the exact alpha entry parents before the failed critic repair, retain307 unchanged, and use a single fixed32-epoch TRAIN-only fit with inherited optimizer, frozen entry tails and no unstated intercept/normalizer/reset changes. Pre-fit coverage reporting must not choose favorable sampler outcomes, change dose based on returns, or select fitting cases by reward. A mechanical/data failure may stop execution; return-range coverage itself should be recorded under the predeclared design. Fresh confirmation worlds must remain untouched until fitting is complete and all analysis choices are frozen.
