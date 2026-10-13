# Independent v5 prelaunch review

Reviewer: `/root/independent_review_astra`, configured as **Astra / high** at the user's explicit request. This report records prelaunch review and independent checks; **the final verdict remains pending** native, retained-PPO, resume and long-panel results.

## Scope and authorization

The previous continuation's failed 73/78 composite gate and exhausted 98,304 additional-native-transition allowance remain intact. The user explicitly authorized a new safe gate and H400 alongside H100. The approved new experiment is 2 critic variants × 2 physical horizons × 2 seeds × 4,096 transitions = **32,768 additional native training transitions**, bringing the cumulative additional allowance to **131,072**, without resetting the old ledger.

Reward remains exclusive to the controlled village. Daughters receive separate native controllers and no daughter welfare is credited to their parent. This review authorizes neither a main merge, GPU training, ecology changes nor unrestricted scale-out.

## Independent design and implementation review

The small-step candidate retains the old critic architecture with learning rate .0001. The other candidate is a new residual-LayerNorm critic:

`V(x) = frozen_original(x) + (trainable_LN(x) - frozen_initial_LN(x))`.

Subtracting identical normalized outputs before adding the teacher preserves original values exactly at entry. This is a normalized adaptive correction, not transparent ordinary LayerNorm insertion and not proof that the frozen saturated teacher has become plastic. Both references are serialized, frozen and excluded from the optimizer. All original named Adam moments, actor/head parameters and episode/RNG state survive conversion; only new normalization parameters begin without prior moments. Collection, training, inference and diagnostics use the composite value. Adaptive, reference and composite health are distinguished.

The 128 construction rows from TRAIN worlds 831001/831002 and 64 validation rows from whole TRAIN world 831003 are separated by world. The analytical transfer performs no fitting or selection on either set. Common-original-value-plus-label targets are retained for all 56 unique original critic targets. H400 intentionally changes the discount/target of an H100-trained warm start; lambda, reward scale and the eight-decision curriculum stay fixed. The existing physical-time return functions correctly change both interval reward integration and `exp(-dt/H)` while retaining truncation bootstrapping.

Independent non-updating restoration of all six original selected checkpoints confirmed exact composite values on all 192 TRAIN rows, immutable/nonoptimized anchors and unchanged original checkpoint digests. Input gradients agreed within floating accumulation error, maximum approximately 9.54e-7 to 1.91e-6; bitwise gradient identity is not claimed.

## Frozen prelaunch decisions and evidence

- **Read-only heuristic benchmark GO:** configuration SHA256 `e4310ad2fa2daeb1325e43edb1176fd150db6adfd62fae454e51b2164b0cde3c`; all 15 input pins independently checked. Approved at most 2,700 recorded transitions plus 2,700 replay resolves, 600 seconds, zero PPO updates.
- **Initial discarded conversion preflight GO:** configuration SHA256 `877b879136199601fd5a4af8d408d40d99b645d79917516bda1681322fb40d0b`; all 13 pins checked. The first launch failed before any transfer/adaptation because a string path was passed to a `Path.read_bytes` API. This was a reviewer-missed implementation bug, not a model failure. The failed output was preserved and a narrow `Path(path)` correction was frozen separately.
- **Corrected conversion preflight PASS:** configuration SHA256 `3c85e5a234076c0cf91125898394c89b5207f3e21bc4fde6add47282f0d95bdb`; all input pins checked after execution. It completed 112/112 retained critic cases, 28,672 critic minibatches in 100.671 seconds, with zero native transitions, sampled actions or retained fitting. All 56 small-step traces exactly reproduce the previous .0001 remedy traces. Both candidates satisfy final and maximum-last-eight MSE < .9 on every target. Worst final / last-eight MSE: small-step .153391 / .260756; residual-LN .440740 / .441749. Maximum composite-forward ratios: 1.056× and 3.568×, below the prospective 4.5× bound. Total model parameter bytes are 2,359,316 and 4,456,476. The residual candidate passes the gate; it does not outperform the smaller-step candidate on synthetic fitting error.
- **Native canary GO:** configuration SHA256 `048b7503b976b27675f8ed8d37cc5c0476800e1ab765032455ace49f00744bb6`; all 32 pins independently verified. Native simulation source digest independently recomputed as `8a1319571e1204a8330306277b96529de23ec3932efff11c61d1964efe996f52`, matching the original baseline. Approved 32,768 training transitions, separate 6,144 development and 6,144 replay resolves, 2,400-second wall limit, four native workers and one Torch thread. Checkpoints at 0/2,048/4,096 retain exact parent entry development comparisons, fresh all-covered-face monitoring, discarded retained critic tests, actual optimizer movement, immutable anchors, resource checks and deterministic failure repeats. Economic score is not an acceptance veto.
- **Postflight GO:** configuration SHA256 `b9680cd0caa6c3a2daee01e21a44dbfc66b06f8d20a7f166b3911409f0fab7b2`; all 13 pins independently verified. Approved deduplicated maximum 66 retained-PPO tasks / 135,168 discarded draws / 4,200 seconds; maximum 384 discarded native validation transitions plus 384 replay resolves / 600 seconds; and a 3,200-transition plus 3,200-replay 25-decision inference panel / 600 seconds. The 384+384 validation ceiling is shared with any independent native reruns; the full primary verifier must not be automatically repeated.

Retained actor coverage comprises four core tasks on every final/stopped candidate, all 13 covered tasks on the worst-confidence checkpoint per critic variant, and four core tasks on the worst actor-saturation checkpoint per variant. Checkpoint/task pairs are deduplicated. This includes technically selected intermediate checkpoints, but does **not** establish all 13 retained tasks on every final model. Fresh-state recovery and retained-Adam adaptation remain distinct.

Independent focused tests rerun during review: 4 model/conversion/checkpoint/H tests passed; later the combined 7 model/budget/selection/variance tests passed; 2 postflight selection tests and 1 food-slot regression also passed. These are overlapping focused suites, not an additive total of independent tests. The root-reported broader suite is separate evidence.

## Defects and qualifications resolved before affected launches

The review requested explicit recorded-versus-replay accounting; actual-versus-planned episode time; conventional return-variance diagnostics with a near-constant flag; actual saved stopped-state selection; and a hard CPU-inclusive resume wall deadline. These were incorporated before the relevant frozen launches.

The draft long panel incorrectly labeled `Stocks[0]` as food. Native registry index 0 is cloth and food is index 3. The new panel uses a named food index with a distinct-value regression test. The same error exists in preserved historical 25-turn evidence, which now has an explicit erratum. That old field cannot support food-stock or harvest claims; local reward, population, QoL, food-needs fulfillment and replay were unaffected.

The first resume draft configured four workers but created only two eight-step jobs. It was changed to four matched four-step TRAIN jobs, exercising all four lanes within the same transition limit. Prefix claims were narrowed to all available original eight-decision summaries plus final fingerprint: unavailable intermediate original action/population/QoL traces are not claimed to have been compared.

## Heuristic evidence and remaining action limitations

The completed heuristic benchmark used 2,656 trajectory transitions plus 2,656 replay resolves, 87.04 seconds, zero PPO updates. I independently recomputed local-reward arithmetic, initial-population normalization and equal-family aggregation on all 192 episode rows and checked all input pins. Native DemandControl exceeds retained model seeds 29/43 by 22.86%/55.60% at eight decisions and 48.98%/148.98% at 25 decisions. The matched public-market-peer arm also exceeds the prospective 20% aggregate threshold. Every full-native family mean exceeds its learned comparator, though some family gains are below 20%.

This is substantial descriptive headroom on this fixed development bank. Native and autonomous trading interfaces/counterparties differ, so it does not causally isolate learned workforce allocation. Native deterministic repeats are not independent sampler replications; episode denominators must accompany trade totals. Daughter welfare remains excluded.

Current cultivation controls allocate farmer labor and expansion effort; native logic still chooses actual cultivation hexes. The retained-herd fraction compiles to a target no larger than the opening domestic herd, and postbreeding excess is slaughtered. This is a structural restriction on retaining net herd growth, beyond mere lack of husbandry training examples. These limitations, separate learned daughter lifecycle, reciprocal learned peers and rare/advanced action faces exclude husbandry/all-domain autonomy or scale-out claims. They require a separate versioned action-contract and curriculum gate, not reinterpretation of retained checkpoints.

**Prelaunch verdict: GO only for the explicitly frozen bounded experiments above. Final scoped readiness remains pending complete raw results and independent final review.**
