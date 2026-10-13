# Independent review of discarded critic investigations

Reviewer: `/root/independent_review`, separate agent context. These are compact
records of the review messages; they are not a replacement for the raw evidence.

The reviewer independently attributed seed29/20,480 recipe-4/9 failures to one
identical critic target. Both actors pass; critic error initially falls, then
oscillates. The frozen continuation remains NO-GO and its native sample envelope
is exhausted. Approval covers the following discarded CPU diagnostics only.

1. **Learning-rate attribution:** GO after preserving partial measurements on
   errors/timeouts and saving a mismatched baseline before rejecting it. The
   implementation now preserves that evidence. Require exact original trace
   reproduction before comparing .0001/.00003 against .0003. No native/action
   draws, retained writes or threshold changes. The reviewer verified exact
   64-point baseline reproduction and lower-rate stabilization after completion.
2. **Read-only critic LayerNorm audit:** GO for six pinned checkpoints and the
   frozen TRAIN probe. Identity-affine LN before both critic tanhs; preserve
   named parameters, actor/heads, original Adam and checkpoint bytes. Measure
   value disruption, ordering and representation health. No optimizer is
   attached to the candidate and no policy is saved. Lower saturation alone
   does not establish usefulness or scale-out readiness.
3. **Matched critic remedy comparison:** code/design preflight passed. The
   reviewer independently reran all three conversion/evidence tests. Execution
   is permitted only after all six original diagnostics complete and config
   pins all raw reports, checkpoints, probe, runner and imported helpers. Use
   common ORIGINAL value-plus-label targets, exact original trace reproduction,
   named retained moments, and new state only for LN affine parameters. Fixed
   variants are .0003, .0001, .00003 and LN+.0001. Prospective final AND last-eight
   MSE must be below .9. Cap 300 seconds and 79,872 critic minibatches with zero
   actor updates, sampled actions or native steps. A failure of LN conversion
   would not prove LN cannot help a differently initialized or longer-trained
   model. Success identifies a candidate, not native scale-out readiness.

The reviewer also verified welfare weighting/decomposition, matched 25-turn
rates and survival caveats, trade-utility-zero semantics, native-versus-PPO
performance distinctions and metadata-only GCS reuse. Two wording corrections
were accepted in the report. The final gate must incorporate all six diagnostics
and the completed remedy comparison; the first target's success is insufficient.

User clarification: rewards remain exclusively village-local. Daughter Elders
must have separate observations, rewards and trajectories; no parent credit for
daughter welfare and no lineage-based reward proposal.
