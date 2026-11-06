# Escalation gate: Fable architecture audit of the r5 critic

Prepared 28 Oct. Launch it only if the trigger below fires. It is read-only, run at model "fable", effort "high". Its findings go to the user; any overhaul it recommends needs the user's decision before work starts.

## Trigger

Evaluate on pilots F1 and F2: separate QoL encoder, trained on fit8-b01, validated on fit8-b04, judged under gate v2 with the r5 reference scale. Launch the audit if ANY of these holds:

1. **QoL underfits even with its own encoder.** In-sample n-step nMAE above 0.30 in both arms.
2. **QoL is far from the gate.** Held-out (b04) n-step nMAE above 0.40, which is 2× the gate, in both arms.
3. **QoL is biased.** The tail-cut residual critic bias, after tail attribution, exceeds 0.10 in the held-out "all" stratum.
4. **Population-years fails for a reason other than tail noise or the old unit.** In the r5 unit, the n-step nMAE stays above 0.20, or the residual critic bias stays above 0.10, OR the population_tail failure is widespread rather than concentrated.
5. **The fit becomes unstable.** Any live divergence alarm under TD rule v2, or any rollback.

## Evidence package

- **Pilot outputs:** gs://anthropocene-rl-datasets-508912/model9/r5/pilot/out-a2, out-d1, out-d2, out-b, out-e1..e5, out-v2 (the re-evaluations) and out-f1/out-f2.
- **Code:** feature/r5-integration at 821660b or later.
- **Decision record:** ops/r5-runbook `ops/r5/decisions/`, plus this worktree's G0-manifest-draft.md and RECALL.md.
- **Previous Fable review:** the starvation, tail-factor and entropy-floor findings are summarized in RECALL.md, entries around 21:30Z on 27 Oct.

## Questions

1. **Is the target design sound for this data?** It combines:
   - a bounded logistic QoL head trained with BCE on TD(λ) targets;
   - a symlog two-hot head for population-years trained on Monte Carlo targets;
   - per-world roulette tails.

   Which component limits accuracy: target noise, loss geometry, capacity or data?
2. **Memorization of world identity.** Do the privileged tokens let the critic memorize which world it is in, and with it each world's target noise? Would masking or regularizing identity-revealing privileged fields help?
3. **Data sufficiency.** The pilots used 8 worlds; the full run has 60. Estimate the error expected at 60 worlds from the 8-to-16-world comparison (D1 versus B).
4. **Should the r5 tail redesign go ahead now?** That redesign (item 4, on hold) stores per-step rewards and draws independent stops per camp. Quantify the share of the remaining error that it would remove.
5. **Is there a case for recurrence or depth?** That is, a GRU or more layers. Cite evidence either way.
6. **Ranked recommendations.** Separate tweaks to the existing structure from overhauls, giving each its expected effect and a cheap pilot test.

The final report should be about 1,200 words or less, quantitative, with file:line references. Mark anything unverified.

## Update 28 Oct ~12:30Z: triggers fired

- **Trigger 3:** F1's held-out tail-cut residual critic bias is −0.172. F2's is −0.097.
- **Trigger 4:** the population_tail failure is widespread, not concentrated.
- Launch is deferred until the saturation audit (g0/saturation-audit.md) and the decision traces (g0/decision-traces.md) finish, so the audit can use both. The user's sequencing is to get those results first.
- New evidence for the audit:
  - pilot/out-v2: gate-v2 re-evaluation of e1, e3, e5, d1 and b. Most of the earlier failure was MC-tail noise plus stale scales. QoL n-step nMAE: E5 0.170, E1 0.183. Population n-step nMAE: 0.08-0.09.
  - pilot/out-f1, out-f2: the separate encoder. Dominance and TD-v2 now pass. QoL n-step nMAE is 0.17-0.18, no better than E5, but the QoL tail-cut bias is now NEGATIVE: residual −0.10 to −0.17, against +0.03 to +0.05 with the shared encoder at weight 30-100.
  - r5 scale artifact: QoL 55.16; population std 135,589.
- Questions sharpened:
  - (a) Why does the separate encoder push QoL negative? Candidates: under-training of the QoL encoder under global-norm clipping dominated by population-years, the Adam/eps regime, or the loss geometry.
  - (b) The population head leaves widespread mass in the top bins (245-254, beyond the observed max of 8.4 scales, against a support top of 11.18). Is the support too wide, is it two-hot smoothing, or is it under-training?
  - (c) The early-turn stratum sits at nMAE 0.20-0.22.
  - (d) E5 (shared encoder, weight 100) versus F2: which should be the baseline for the scale-out?
