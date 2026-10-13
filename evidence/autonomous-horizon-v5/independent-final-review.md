# Independent v5 final review

Reviewer: `/root/independent_review_astra`, **Astra / high**, explicitly requested by the user.

**Verdict: PASS for the completed, frozen mechanical experiment; NO-GO for further training promotion or economic scale-out.** All prospective mechanical tests completed successfully. A separately discovered and independently confirmed observation-history defect means this campaign did not test the intended MLP with persistent institutional memory and autonomous previous-action evidence. Repair and version that contract before further promotion. The documented GRU comparison gate has not been reached.

The previous continuation's 73/78 composite result remains a failure in the historical record. Neither this experiment nor the new residual architecture retroactively changes it. The new 32,768-transition training allowance is now exhausted; cumulative additional native training is 131,072, with the original pilot separately recorded. Proposed follow-up experiments require separate frozen budgets.

## Verified mechanics and accounting

I reviewed the design and concrete code before the respective launches and checked the frozen input pins. The primary native gate configuration is `048b7503b976b27675f8ed8d37cc5c0476800e1ab765032455ace49f00744bb6`; postflight configuration is `b9680cd0caa6c3a2daee01e21a44dbfc66b06f8d20a7f166b3911409f0fab7b2`. All 32 native, 13 postflight and five corrected-panel pins remain intact after completion. The native source digest independently recomputes to the unchanged `8a1319571e1204a8330306277b96529de23ec3932efff11c61d1964efe996f52`.

The two critic candidates both use LR .0001. Small-step preserves the old architecture. Residual LayerNorm computes

`V(x) = frozen_original(x) + (trainable_LN(x) - frozen_initial_LN(x))`.

The subtraction occurs first, so conversion preserves original entry values. The normalized correction is trainable; both serialized reference networks stay frozen and outside the optimizer. This is not ordinary LayerNorm insertion and does not make the original saturated teacher plastic. Original named Adam moments, actor/head parameters and RNG/episode state are preserved. All inference, bootstrap and training paths use the composite value.

Independent restoration on all six original selected checkpoints confirmed exact values on all 192 TRAIN rows, frozen/nonoptimized anchors and unchanged checkpoint digests. The analytical transfer uses 128 rows from TRAIN worlds 831001/831002 and 64 validation rows from whole TRAIN world 831003, with no fitting. The corrected preflight completed **112/112 retained critic cases** on common original value-plus-label targets, 28,672 critic minibatches, 100.671 seconds, zero native steps or action draws. All 56 small-step traces exactly reproduce the earlier .0001 remedy traces. Worst final / last-eight MSE was .153391 / .260756 for small-step and .440740 / .441749 for residual-LN. Maximum composite-forward overhead was 3.568×, below the prospective 4.5× limit.

The fixed 2 critic × 2 horizon × 2 seed native experiment completed **eight cells × 4,096 transitions**, with 64 actual PPO updates and 512 critic minibatches per cell. All **24/24** entry/2,048/4,096 boundary fresh-recovery and retained-critic checks passed. It used 1,610.272 seconds, four native CPU workers and one Torch thread. Separate development/replay use was 6,144 + 6,144, so this campaign made 45,056 native resolves. No in-flight reservation remains.

I independently restored **all 24 saved boundaries** and verified their digests and model hashes, all eight entry actor/value/named-Adam/RNG equivalences, finite optimizer tensors, immutable anchors outside the optimizer, actual nonzero actor/critic steps, and final critic Adam step increments of exactly 512. This review performed no additional native transitions for those checks.

The retained-PPO postflight completed **66/66 actor and composite tasks across 12 selected checkpoints**, consuming exactly 135,168 discarded action draws in 2,880.167 seconds. I independently checked every 64-update trace, exposure threshold, final actor separation/gain, final and maximum-last-eight critic error, checkpoint/task selection, deduplication and source/checkpoint hash. Worst final / last-eight MSE was **.227588 / .295443**, both below .9; minimum actor separation gain was **.095209**, above .01.

Coverage was the frozen union: four core tasks on all eight final candidates; all 13 covered tasks at the worst-confidence checkpoint per critic variant; and four core tasks at the worst actor-saturation checkpoint per variant. Technical selections include intermediate checkpoints. This is **not** all 13 retained tasks on every final model. Fresh-state recovery and retained-Adam adaptation are distinct evidence.

The primary native resume verifier passed on all 12 selected checkpoints: requested actions/probabilities, replay, **four active lanes** versus one, next PPO model/Adam/RNG state, frozen anchors and composite inference equivalence. I audited the code and completed raw report/pins against the exact selected set. It consumed the full shared 384 discarded validation transitions + 384 replay resolves in 50.250 seconds. I did **not** repeat native simulations and exceed that allowance. Independent focused model, budget/selection, variance, food-slot and sequential-summation regression checks also passed.

## Economic results do not constitute promotion

The corrected 25-decision panel completed 128 episodes / 3,200 trajectory transitions plus 3,200 replay resolves, with zero PPO updates. I independently checked every episode's local reward arithmetic, all available original eight-turn summary/final-fingerprint comparisons, physical H100/H400 discount integrals, equal-family aggregates and checkpoint pins. The prior failed reporter's 25 + 25 resolves remain charged; inclusive panel use is **3,225 + 3,225**. Original intermediate eight-turn action/population/QoL traces were not saved, so their independent comparison is not claimed.

H400 has lower 25-decision displayed welfare than H100 in all four matched critic/seed comparisons. It also has much larger native target error: mean preupdate MSE/return variance 1.364–2.066 versus .223–.311 at H100. This is evidence against promoting this short H400 warm-start configuration, not proof that H100 is universally optimal. Two training seeds do not establish a robust architecture/horizon ranking.

Small-step H400 critic saturation reaches 96.5% while its current retained probes still pass. Residual-LN maintains a largely unsaturated adaptive correction, with final affine scale maxima below 1.058, while the original teacher stays saturated. These observations distinguish local retained responsiveness, native prediction accuracy and economic competence; none substitutes for the others.

Every candidate has survival-family panel episodes ending with one to three residents, and all 128 panel episodes settle zero focal trades. Population losses include migration and fission and are not all deaths. A mechanically responsive policy is not thereby an economically strong one.

The separate native-heuristic audit supports substantial descriptive headroom. On all 16 development snapshots, native DemandControl exceeds the retained v3 parents by 22.86%/55.60% at eight decisions and 48.98%/148.98% at 25. The public-market-peer comparison also meets its prospective aggregate threshold. I independently recomputed local reward and equal-family normalization for all 192 reported episode rows.

On the matched eight-ID 25-turn panel, native raw welfare is 166,726.38 and its displayed score is 1,121.25. All eight new models have **lower raw local welfare in every family**, 32/32 model-family comparisons, with deficits 34.78%–81.25%. Thus the economic gap is not merely an artifact of displayed initial-population normalization. Native trading/action interfaces differ and deterministic native repeats are not independent sampler replications; the result does not isolate neural allocation quality causally.

## Blocking observation-history finding

The additional independent training assessment identified a major contract gap, which I independently reproduced with source inspection and existing data only:

- All 192 saved TRAIN probe rows have zero previous-decision channels 40–55 and zero memory maturity channels 117–119.
- Each of the eleven nominal EMA signals has exactly identical 20/60/180-year channels in slots 84–116.
- All 28 bank saves contain four `ElderMemory` binary serialization markers, each nil. This inspection is of decompressed binary markers, not an additional native restore.

The source chain explains it. `integration/SnapshotScenarioFactory.cs:298` clears memory. `AutonomousElderEnvironment.Reset` never initializes it. `ElderObservationBuilder.cs:50–53` therefore obtains a new `TransientView`, rather than persisted history. `AutonomousElderCompiler.Compile` returns no Elder decision receipt, and the autonomous early return in `TurnPlanning.cs:330–331` bypasses that recording path. `TurnResolver.cs:203–204` advances only nonnull memories. Existing previous-decision fields describe the earlier supervisory action interface, not the current workforce/recipe/quote actions.

The authoritative architecture snapshot, `runs/drive-architecture-gate-v5.txt`, §§39.7/39.10/41.6, requires persistent institutional traces and legitimate previous-action/outcome evidence before concluding that the feed-forward baseline lacks memory capacity. The current path does not satisfy that prerequisite. This does not prove the omission caused all welfare loss, nor that a GRU would solve it. It means **neither MLP sufficiency nor qualifying MLP insufficiency has been demonstrated**, and GRU comparison/promotion is premature.

A repair must be versioned and preserve the frozen evidence. It needs actual autonomous action/outcome receipts, validity and physical ages, persistent local EMA semantics, distinct-history checks, save/load/replay and partition-safe ablations. Merely allocating memory would leave the action-history gap unresolved; stuffing current actions into obsolete five-head `ElderDirective` fields would misrepresent their meaning. The observation/history repair precedes further architecture or scale-out claims.

## Preserved failures and remaining scope

The first conversion preflight failed before adaptation because a string path reached a `Path.read_bytes` API. Its failed report/config/source history remain preserved; the narrow corrected preflight passed without changing thresholds. I missed that type mismatch in the first static review and did not misclassify it as a learning failure.

The first long panel completed and replayed one trajectory, then failed exact scalar comparison because Python 3.12's compensated `sum` differed from the native collector's sequential `+=`. The independently reproduced normalized difference is approximately −1.82e-14; its fingerprint matched. A separately pinned fixed runner retained exact fingerprint/summary checks, matched the original accumulation order and persisted raw rows before assertions. The extra 25 + 25 resolves were prospectively charged, not erased.

The draft panel also inherited an old reporting bug: stock index 0 is cloth, while food is index 3. The new panel uses a named food index and regression test; old evidence is preserved with an explicit erratum. Those historical mislabeled fields support no food-reserve claims. Reward, QoL, population and food-fulfillment evidence are unaffected. Food stocks still are not direct harvest measurements, and field-support aliases are not forecasts.

Rewards remain exclusively village-local. Daughters need separate learned observations/rewards/trajectories; no dynasty or lineage reward is recommended. Current daughters still use native heuristics. Cultivation hex selection remains native, and herd retention targets cannot exceed the opening herd, so postbreeding excess is slaughtered. These are structural control restrictions beyond absent husbandry/advanced-face examples. Reciprocal learned trade, rare action faces and untouched regime holdouts also remain missing. No hidden inventory, q_k shortcut or prospective planner-value oracle is justified by the current failures.

**Final decision:** retain the successful mechanical evidence and all checkpoints, but do not promote H400, broaden training, scale out, or replace the MLP with a GRU on this evidence. First repair and validate the versioned autonomous history contract, then use separately bounded actual-target critic/coverage and economic-adaptation comparisons. No main merge, GPU/ecology work or unbounded continuation is approved.

Independent auxiliary artifacts: `/tmp/autonomous-horizon-v5-independent-preflight-review.md`, `/tmp/autonomous-horizon-v5-independent-restore-audit.json`, `/tmp/autonomous-horizon-v5-independent-panel-audit.json`, and `/tmp/autonomous-horizon-v5-independent-retained-audit.json`. The separate causal assessment and its reproducible read-only audit contain the detailed GRU clause mapping.
