# Autonomous Village Elder PPO: independent causal assessment and GRU gate review

13 October 2026. Independent Astra/high assessment, based on source inspection and existing evidence only. This review performed **zero native simulation steps, zero PPO updates and zero sampled actions**. It does not authorize a new campaign or implement a recurrent model.

**Conclusion:** The results are economically weak, but they do not establish that the feed-forward architecture has reached its limit. A concrete observation/history plumbing defect, a very small and short curriculum, a measured H400 critic-target mismatch, and incomplete trade/action interfaces all precede recurrence in the causal investigation. The Google Drive GRU gates have **not** been reached. In particular, the autonomous actor has not actually been tested with the persistent institutional memory and previous-action evidence that the gate requires.

## 1. What is established

The completed V5 experiment contains eight cells: two critic variants × H100/H400 × seeds 29/43, with 4,096 additional native training transitions per cell. All 24 boundary fresh/retained critic checks pass. Exact entry-value preservation and immutable residual anchors support the interpretation that the critic conversion is mechanically sound. These checks establish bounded adaptation and reproducibility, not economic competence or global representation sufficiency.

On the matched eight-snapshot, 25-decision panel, the native heuristic's mean raw village welfare is 166,726 versus 51,631–89,249 for the learned policies. The displayed population-normalized score is 1,121 versus 273–627. All eight learned policies underperform the native baseline on **each family's mean raw welfare**, not just on the displayed normalized score:

| Family | Native mean raw welfare | Learned range | Learned deficit range |
|---|---:|---:|---:|
| Fission | 213,961 | 67,756–139,541 | 34.8–68.3% |
| Migration | 188,191 | 35,282–116,296 | 38.2–81.3% |
| Survival | 51,755 | 10,794–19,901 | 61.5–79.1% |
| Trade | 212,999 | 61,676–118,232 | 44.5–71.0% |

These are descriptive comparisons on the same eight reset IDs. The native run has one deterministic rollout per reset; each learned model has two stochastic samplers. Native and autonomous trade/action interfaces differ, so this is evidence of headroom, not a controlled proof that an equally informed MLP could reproduce every native action.

Every model has survival-family trajectories ending with 1–3 residents. All 128 learned 25-decision episodes settle zero focal trades. Some population losses include fission and emigration; they must not all be labeled deaths. The 25-turn welfare gap is substantially a population/person-year gap: native time-mean population is 371.9, versus 122.5–205.3 for learned policies, while person-year-weighted QoL is approximately .897 versus .819–.910. High QoL among remaining residents does not establish a healthy village trajectory.

## 2. Evidence-ranked causal assessment

### A. Missing intended history inputs: confirmed defect; contribution to welfare unmeasured

The strongest newly identified contract issue is that autonomous observations retain the *slots* for institutional history without supplying the intended persistent history.

Independent read-only inspection finds:

- All 192 saved TRAIN probe observations have **zero previous-decision fields in slots 40–55**.
- All 192 have **zero memory maturity in slots 117–119**.
- For every one of the eleven signals, the 20/60/180-year channels in slots 84–116 are **exactly identical**.
- All 28 bank snapshots contain four camp `ElderMemory` serialization markers, all nil. This last check inspected decompressed binary markers; it was not an additional native restore.

The source explains these observations. `SnapshotScenarioFactory.cs:298` sets camp memory to null. `AutonomousElderEnvironment.Reset` does not initialize it. `ElderObservationBuilder.cs:50–53` therefore creates a new transient view at each observation, rather than reading a persisted EMA history. Boundary integration in `TurnResolver.cs:203–204` visits only camps with nonnull memory. Separately, `AutonomousElderCompiler.Compile` returns an `AutomaticPlan` without an Elder decision; `TurnPlanning.cs:330–331` takes that autonomous early return, so the legacy `RecordDecision` path has no autonomous action receipt to record. Existing prior-action fields describe the older supervisory `ElderDirective`, not the current workforce/recipe/quote action contract.

Consequently the actor sees current-state signal copies in the nominal EMA channels, not useful multi-timescale trend discrimination. This is **not** evidence that correctly implemented deterministic memory is insufficient. A future fix must record actual autonomous requests and outcomes, with explicit validity and physical ages, instead of pretending the obsolete five-head previous-action layout already represents the new actions. Simply allocating an EMA object would leave the autonomous action-history problem unresolved.

This defect is independently confirmed by the other reviewer. Its existence is certain on the audited path; the amount of economic loss it causes is not known without a controlled ablation. A policy can still learn useful reactive behavior without history, so it would be excessive to attribute all failures to this issue.

### B. Narrow resets, sparse successes and inadequate duration coverage: strong training-design concern

The bank contains **12 TRAIN snapshots from three worlds**, covering four families, and **16 DEVELOPMENT snapshots from four different worlds**. Whole-world separation is good. Repeating these resets for many transitions does not create additional independent starting conditions, historical regimes, or held-out worlds. There is no untouched held-out regime cohort in this bank.

Training resets after eight decisions, or 160 years; long evaluation runs for 25 decisions, or 500 years. Later capital depletion, repeated movement, post-fission recovery and low-population states are therefore weakly covered or absent in training. The current bank lacks the documented institutional-memory contrast and controlled frontier-change curriculum, and does not establish husbandry, fishing or advanced-recipe competence. More epochs on the same bank would not resolve those coverage gaps.

This is not a claim that eight-step training wrongly treats truncation as death: the collector correctly bootstraps value at truncation and stops the GAE carry across resets. The problem is reliance on a learned estimate beyond a systematically short visited horizon. Nor does it prove that merely increasing reset length will succeed. Equal-budget short-versus-long training is the discriminating comparison.

Trade training provides exceptionally sparse success: across 512 episodes per cell, six cells settle no focal trade and the other two settle only one and two. A high-entropy, multi-component quote policy receives little evidence linking a particular partner/good/price combination to welfare. That favors focused legal trade opportunity coverage and interface diagnostics before increasing network memory.

### C. H400 critic underfitting and target shift: directly measured, particularly important for the H comparison

H400 is a warm start from an H100 value function, not an equally trained alternative horizon. Mean preupdate MSE divided by actual return variance is **1.364–2.066 at H400**, versus **.223–.311 at H100**. H400 value bias averages approximately **−36 to −51 scaled units**, versus roughly −1.1 to −.08 at H100. Synthetic residual recovery can pass while native on-policy targets remain badly underpredicted.

An eight-decision bootstrap has coefficient exp(−160/H): **.6703 at H400 versus .2019 at H100** before GAE weighting. With lambda .95 per 20-year decision, the combined GAE trace e-folding time is approximately 197 years at H400 and 80 years at H100. H400 therefore depends more strongly on tail estimates precisely when its critic is adapting to a larger target. Correct physical discount arithmetic does not make those estimates accurate.

H400 has lower 25-decision displayed welfare in all four matched critic/seed comparisons. That supports a warning about this warm-start/budget configuration, not a universal conclusion that H100 is optimal or that long-horizon welfare is unlearnable. Two seeds and correlated architectural forks are inadequate for a strong statistical ranking. The next test should first isolate actual-target critic adaptation with the actor frozen, then compare equal-budget actor learning with declared critic readiness and longer reset coverage.

### D. Action and counterparty restrictions: confirmed scope limits; causal importance differs by domain

Current controls express workforce shares, conditional recipes, farmer expansion effort, breeding intensity, herd retention, movement and stock quotes. They do not provide unrestricted economic authority.

- **Cultivation:** native code still selects cultivation hexes. This limits explicit spatial autonomy, but it is not yet demonstrated to explain the present baseline gap; native cultivation selection may itself be helpful.
- **Herds:** the target is `floor(opening animals × retained fraction)`, and postbreeding excess is slaughtered. Retention cannot exceed opening herd size. This prevents net offspring retention under the intended growth decision. Moreover, a continuous Beta fraction is almost surely below one, unlike the exact endpoint accepted by native code. This is a serious husbandry control issue, but husbandry is absent from the current bank, so it cannot explain all current cultivation failures.
- **Trade:** `AutonomousElderTrade.Plan` requires the counterpart to be present in the submitted controlled-action dictionary and to submit a reciprocal quote. A known native heuristic contact can pass the visible counterparty mask while being unable to reciprocate through this interface. Survival/fission resets have no scripted controlled peer; trade/migration resets have one. Broad known-contact availability is therefore not equivalent to an executable exchange opportunity. Current contact rows do not contain public offer terms or a counterparty protocol indicator. Hidden remote inventory should remain hidden, but legitimately public executable offers/protocol status warrant a separate observation/interface review.
- **Joint quote choices:** enable, partner, give, want, lot and price must align. Most logits depend only on the shared state representation; conditional masks impose legality but do not provide a fully conditioned partner/good-dependent decoder. This may make heterogeneous counterparties harder, although the fixed single-peer task should be simpler and does not prove decoder insufficiency.
- **Daughters:** newly founded daughters use native heuristic controllers. Their subsequent welfare is excluded, as required by the explicitly village-local objective. This is an incomplete autonomous lifecycle, not a reason to add dynasty reward or silently change the user's objective.

There is also a distributional limitation separate from hidden-layer capacity: continuous concentrations are `softplus(clamp(logit,−12,12))+.5`. On an active two-coordinate control face, the attainable *mean* is bounded near .0385–.9615, and samples retain nonzero dispersion; exact zero/full actions are not ordinary deterministic means. The current 5% uniform categorical mixture similarly imposes continued random legal choices. Neither mechanism is a programming error by itself. Both make deterministic native-versus-stochastic learned comparisons hard to interpret and deserve endpoint/execution-mode diagnostics before attributing the gap to recurrence.

### E. Collapse, saturation and oscillation: distinguish mechanisms

**Broad actor collapse is not supported by the current observations.** Final actor tanh saturation on the saved probe is approximately 0–.36%; the policies retain nontrivial action entropy and observation variation. Parameter changes occur in every actual PPO update. The prior continuation passed 78/78 retained actor tests; current final retained-actor work remains a bounded diagnostic and must be reported at its eventual completed count. Such probes show the model can adapt to their local synthetic targets; they do not establish sufficient capacity for economic control.

**Localized output clamp saturation remains real.** Approximately 2.3–6.9% of active simplex logits are beyond the clamp at final checkpoints. Those logits have zero local gradient through the clamp. This is not the same as whole-network collapse, and successful recovery elsewhere does not prove every economically important endpoint is reachable.

**Critic saturation is real but not sufficient to explain the outcomes.** Small-step H400 has heavily saturated critic layers, up to 96.5%, while still passing retained critic probes. Residual LayerNorm supplies a largely unsaturated adaptive path, yet does not reliably restore high welfare and still shows H400 native-target underfitting. The old teacher's frozen saturation is not evidence that the new adaptive path is stuck.

**Oscillation has historical evidence, but current welfare variation is not a diagnosis of a limit cycle.** The old .0003 critic had retained-target oscillation; .0001 and the residual remedy pass their common target tests. Current entry/mid/final development scores include recoveries and declines. Three measurements per cell do not identify periodic behavior, and stochastic trajectories differ across models. Mean actor minibatches actually applied range from about 4.3 to 5.7 of eight per update under the KL stop, indicating active constrained optimization rather than a total stall. A useful future audit should track fixed-state analytic distributions, native held-out value error, gradient directions and policy turnover over time. More sampled synthetic probes are unnecessary in this completed budget.

### F. Objective and reporting weights: clarify; do not replace the goal

The authorized objective is raw local prior-population × QoL × physical time. PPO uses this, divided by a fixed 120 scale and physically discounted. The displayed score divides each episode by its initial population and averages families equally. These are different summaries, so a change that improves one need not improve the other.

Initial family populations range from 65 to 480, a 7.38-fold range. Global rollout advantage normalization does not remove all effects of heterogeneous reward scale or critic errors. However, realized family rewards are not simply proportional to initial population: migration can greatly change population. In the actual training ledger, average raw eight-turn welfare is roughly 12.4–14.2 thousand in survival versus 22.5–26.5 thousand in trade. A per-family gradient/advantage analysis would determine whether one family dominates updates. None of this justifies silently replacing population-weighted welfare with per-capita reward. The independently calculated raw-welfare deficits above show that the economic failure is not just a reporting-normalization artifact.

## 3. GRU gates: clause-by-clause verdict

Source: [Anthopocene AI/RL Training Concept—Neolithic Curriculum and Scalable Agent Architecture](https://docs.google.com/document/d/1rZjIo7NMrxn3OQzGFHJI2QAZN7rauHgSCm-nbMNUCT0/edit), retrieved snapshot `runs/drive-architecture-gate-v5.txt`; source last modified 10 October 2026. The old 512-feature supervisory action/schema pins are outdated relative to the authorized autonomous 761-feature interface. The following experimental logic still applies; it must be mapped explicitly to current controls rather than using obsolete head labels.

| Source clause | Verdict on current evidence |
|---|---|
| §41.3: distinguish pre-evidence ignorance from post-evidence adaptability | **Not tested by this campaign.** Poor aggregate welfare is not a paired latent-regime experiment. Unobserved remote changes cannot be inferred by any architecture. |
| §41.4: actual-interface fixed q_k=.4/.5/.8 and within-episode regime-change sandbox | **Not demonstrated on the current autonomous contract.** Four economic reset families and H100/H400 are not the required capital-frontier cohorts. Legacy investment fixtures are not interchangeable evidence. |
| §41.4: action-distribution sensitivity after a revealing investment | **Not demonstrated.** No matched pre-/post-evidence adaptation effect and latency measurement on current heads. |
| §41.5: at least three neural seeds and held-out regime/snapshot combinations | **Not met.** Current matched campaign has two seeds, development worlds, and no untouched regime-change held-out cohort. |
| §41.5(1): economically correct post-evidence directional response | **Unknown.** Synthetic recovery tasks do not measure this economic direction. |
| §41.5(2): adaptation without q_k/technology strategy labels | **No such label shortcut is required by the current interface, but successful regime adaptation is unproven.** Structural legality masks alone are not a prohibited strategic recommendation. |
| §41.5(3): welfare regret within predeclared tolerance versus best available legal action | **Not met as an experiment.** The native heuristic is a useful comparator but not a certified best current-interface action oracle, and no sandbox regret tolerance has been tested. |
| §41.5(4): unseen regime value or timing outside checkpoint selection | **Not demonstrated.** |
| §41.5(5): no excessive repairs or hand-engineered response curve; report effect size and physical latency | **Partially compatible mechanics, missing adaptation evidence.** Requested-action likelihood is preserved and poor legal choices are not generally rescued; economic adaptation magnitude and latency are absent. |
| §41.6: MLP fails paired histories despite current state + persistent EMA/event ages + previous action/outcome | **Not reached; baseline prerequisite is broken on this path.** Persistent EMA and autonomous previous-action receipts are missing. No qualifying paired-history failure has been shown. |
| §41.6: useful distinction exists in ordered permitted observation/action history | **Unproven.** Need to show a legally inferable historical difference with different optimal actions, not just a hidden simulator difference. |
| §41.6: matched MLP/GRU partitions, transitions/wall budget and seeds | **Not run.** Neither compare nor promote yet. |
| §41.6: statistically/materially meaningful held-out welfare/adaptation gain unreproducible by observation fixes or a few deterministic traces | **Not met.** No recurrent comparison; a concrete observation fix remains untested. |
| §41.6: separate time-aware actor/critic GRUs, physical Δt, reproducible reset/burn-in and sequential hidden-state continuity | **Future implementation requirements, not satisfied evidence.** If eventually eligible, update the encoder input dimension and current action heads explicitly. |
| §42.7 and §36 temporal pivot: EMA ablation and earlier diagnoses before GRU | **Not met.** A true persistent-memory baseline and ablation are required first; current transient copies are not a meaningful EMA treatment. |
| §41.7: no default prospective response oracle | **Keep deferred.** Current failure does not authorize exposing hidden q_k, planner marginal value or a prospective response curve. First repair legitimate observed histories and measure exploration/aliasing. |

**Decision:** Neither “MLP sufficiency passed” nor “MLP insufficiency requiring recurrence proved” is justified. The specific GRU *comparison* eligibility gate is not reached; the *promotion* gate is necessarily further away.

## 4. Discriminating next experiments, in order

These are proposals for the next separately bounded authorization, not additional work executed by this assessment.

1. **Repair and verify current autonomous history semantics.** Version actual action/outcome receipts and persistent village-local memory. Exercise nonzero maturity, distinct histories with identical current signals, physical-time decay, correct reset/save/load/replay and daughter independence. Preserve current checkpoints and completed evidence. This is a correctness prerequisite, not a neural-capacity change.
2. **Compare fixed MLP with and without the repaired permitted history.** Use matched worlds, legal trajectories, three or more neural seeds and a declared finite budget. Include improving/declining histories with comparable current state. Separate initialization effects from information gains. If the corrected MLP adapts, recurrence is not needed to solve that contrast.
3. **Isolate H400 critic adaptation using actual TRAIN targets.** Retain observation/reward/next-value/return traces; freeze actor and preserve its parameters, Adam state and RNG. Fit only the critic within a fixed budget and validate on world-disjoint TRAIN trajectories. Compare calibrated and uncalibrated starts before interpreting H400 actor behavior. Report native value bias/error by family and rollout depth, not only synthetic residual MSE.
4. **Compare eight- versus 25-decision training at equal native budgets.** Keep architecture, objective, interfaces and development selection rules fixed. Keep H100/H400 as declared arms if affordable; do not collapse horizon and reset length into one change. Report tail population, QoL, annual welfare rate, food reserves, field retention and migration/fission separately. More late-state coverage should help if reset bias is a major cause.
5. **Measure action/interface headroom without changing the reward.** Compare stochastic deployment with the model's defined deterministic mean/argmax mode, and quantify endpoint/continuous concentration limitations. On TRAIN trade fixtures, record proposed quote, available public counterpart protocol, exact rejection reason and settlement. Where possible compare native actions projected into the exact current legal interface. These distinguish search failure, inaccessible actions and informational deficits from a memory problem.
6. **Broaden achieved curriculum coverage, not just snapshot count.** Add legal capital-maintenance/reserve contrasts, actual trade opportunities, post-fission states, and mature history. Version herd-growth retention and cultivation-hex authority before claiming those domains are learned. Add husbandry and later capabilities only with their correct action/observation contract. Preserve whole-world/lineage/counterfactual sibling partitions; introduce genuinely untouched held-out worlds and frontier timing.
7. **Run the §41 regime sandbox and paired-history test.** First establish post-evidence MLP regret and directional adaptation against a legal current-interface reference with predeclared tolerances. Only if decision-relevant histories remain aliased after legitimate deterministic memory fixes should a small time-aware GRU receive a matched-budget comparison. Promote it only on the document's held-out material-gain criterion.

The immediate priority is history correctness plus measured critic/coverage diagnosis. A GRU can remember information that arrived; it cannot supply missing current public protocol data, create unavailable herd-growth actions, manufacture reward examples, or make an underfit value target correct.

## 5. Evidence, provenance and limits

Primary independent audit: `/tmp/autonomous-elder-training-assessment-astra-audit.json`; reproducible read-only script: `/tmp/autonomous-elder-training-assessment-astra-audit.py`. The JSON hashes its inputs and records memory counts, source digests, family raw welfare, training critic statistics and horizon arithmetic. It imports no environment, samples no actions and runs no optimizer.

Primary experiment artifacts: `runs/autonomous-critic-horizon-gate-v5/summary.json`, `runs/autonomous-horizon-welfare-v5-fixed/summary.json` and per-model episode files, `runs/autonomous-heuristic-welfare-v1/summary.json`, `runs/autonomous-canary-preflight-v1/probe-v2.pt`, and `datasets/autonomous-ppo-gate-v1/manifest.json`. Source references are in the audit manifest. The erroneous historical food-stock label is not used; the new panel correctly measures food at index 3. The fixed panel's original arithmetic failure and charged replay retry remain preserved.

At this review's source snapshot, RL HEAD was `a3796a8e540102ea1011a271c7ec5bdce99c6193`; native HEAD was `65153f7d1e7bbae880b55650d34431d4814e8243`. These are review-time repository heads, not a replacement for the campaign's pinned code/source digests or final delivery commits. Native campaign source SHA256 is `8a1319571e1204a8330306277b96529de23ec3932efff11c61d1964efe996f52`.

The retained actor diagnostic suite was still running when the causal audit was assembled; final publication should append its actual completion count. Its outcome cannot repair the missing history channels or establish the GRU gate. The assessment does not apportion economic losses numerically among causes: that requires interventions. Development worlds, repeated samplers and architecture forks are not independent statistical replications, and repeated consultation of development evidence prevents treating it as pristine held-out proof.

## Publication verification
The retained diagnostics completed after the causal assessment above was assembled: all 66 actor/composite tasks passed across 12 selected checkpoints, using exactly 135,168 sampled diagnostic draws in 2,880.167 seconds. Each task used 64 retained PPO updates with 32 samples. Worst final/maximum-last-eight composite MSE was 0.227588/0.295443; minimum actor gain was 0.095209 against a 0.01 criterion. Coverage consists of four core tasks on all eight final models plus full 13-task coverage at predeclared selected extrema; it does not mean all 13 tasks were run on every final model.

The final implementation reviewer, independently using Astra/high, also verified the traces, checkpoint restoration, source pins, exact replay and budget accounting. Its scoped verdict is PASS for the completed frozen mechanical experiment and NO-GO for further training promotion or scale-out until the institutional-history/action-receipt contract is repaired and retested. Twelve selected checkpoints passed exact one-worker versus four-worker continuation and inference checks, charging 384 discarded validation plus 384 replay steps to the existing cap.

These completed mechanical results do not change this assessment's causal conclusions: the missing history inputs remain a confirmed correctness defect; their share of the welfare loss remains unmeasured; and the GRU comparison and promotion prerequisites have not been met. No GRU, new native memory fix, additional training campaign or scale-out was executed for this assessment.

Google Doc: https://docs.google.com/document/d/1tgk3YC4R-10axzjRA9tQuRURjsnsbqD9m5OZMOV9W04/edit

