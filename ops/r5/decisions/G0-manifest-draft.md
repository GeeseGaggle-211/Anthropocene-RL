# G0 campaign manifest — DRAFT (labor actor, v1.9 brief)

Status: **draft, not frozen.** It freezes when the G1 branches are merged, their commits pinned below, and the measured throughput fills the budget table. Reviewers (both roles): Claude (co-architect), per the user's 24 Oct direction. Implementation goes to Opus/Sonnet subagents.

## 1. Identity (filled at freeze)
| Field | Value |
|---|---|
| Learner | GeeseGaggle-211/Anthropocene-RL, merge of feature/g1-fixes + feature/labor-film onto feature/r4-economic-observability (87b99d4) — commit at freeze |
| Simulator | GeeseGaggle-211/Anthropocene, feature/g1-fixes onto 4c52da7 — commit at freeze |
| Bridge binary | hash recorded by the VM build (`~/git/checkout.sh`) |
| Contracts | actorLabor 13; action 12; observation/environment/replay 13; normalizer model9-bounded-physical-history-v1 (admitted checkpointed normalizer); input contract model9-bounded-labor-feedback-v1 |
| Rules | new-world defaults: PhysicalMortalityV1, SurplusFundedBirths, ElderKernelSplit, ElderMixedRoutes, SettlingGraceOnce, PlantTentsEnabled; 20-year decisions |

## 2. Objective (unchanged)
Task components: lineage QoL-years and population-years, with frozen reference scales. Costs: feasibility and settlement guard, multipliers 1.0/1.0. Discount time constant 400 years. **No capital, clothing or housing bonus.** The actor is expected to discover the value of clothing and housing through their effect on mortality and growth (user, 24 Oct). Gross turnover is reported, not penalised.

## 3. Personalities (added per user)
- Architecture: zero-initialised, bias-free FiLM, conditioned on c = p − (0.5, 0.5), in the actor and all three critic encoders (feature/labor-film).
- Fitting distribution: Dirichlet(1, 1) over (q, p), drawn once per founding lineage per world from a keyed stream. Daughters inherit with Dirichlet drift at concentration 200.
- Evaluation grid: balanced (0.5/0.5), plus (0.9/0.1) and (0.1/0.9), reported separately.
- Known limit: heuristic BC labels carry no personality signal, so G2A teaches personality-invariant behaviour. Personality-specific behaviour is expected only from G3 search labels scored per personality and from PPO. All four critics are fitted at every personality, including a component whose weight is zero.

## 4. World partition (fresh bank)
| Role | Worlds |
|---|---|
| Fitting (80) | 90100–90179 |
| Critic calibration (20) | 90180–90199 |
| Development (historical reuse disclosed) | 2000–2059, never fitted |
| Confirmation (sealed) | 3000–3059 |

All snapshots, descendants and branches stay in their world's partition. The collector must check this range against every historical seed list before generating, and refuse to run on any overlap.

## 5. Interim pretraining candidate (tonight) — scope and declared deviations
| Item | Brief | Tonight | Status consequence |
|---|---|---|---|
| Teacher | H12 via `query_action12_teacher_v1` plus planned-job ledger | Corrected converter on native H diagnostics, with the Herder ledger, idle preservation, quarantine, and missing → unavailable | G1 teacher-fidelity gate HOLD until H12 meets 95/90% of H |
| BC objective | Mean CE + 4 sampled draws, stable CE | Stable masked CE; existing dispersion regulariser declared as the alternative | Declared deviation |
| Schedule | batch 256, peak LR 3e-5, 10% warm-up, cosine to 10%, gradient cap 1, 50 observer epochs | Same | — |
| Data | Full-context roots; chronological Standard-80 lane | 80 fitting worlds × 25 decisions, movement on, world-balanced | Standard-80 chronology deferred, so the chronological competence gate HOLDs |
| Critic | V^rho, four heads, independent tails, pilot of 16 → production 16/64 streams | Same producer (labor_anchor_producer randomised tails) on 80 fitting and 20 calibration worlds; 100 epochs | G2C numeric criteria apply as written |

The candidate is an **initializer**, not a gate pass. Its receipts record exactly which gates are PASS, HOLD or FAIL.

## 6. Acceptance checks run tonight
- **Actor:** support ledger per category (≥32 fitting roots across ≥4 worlds; ≥16 development roots across ≥4 worlds); same-state specialty and activity error on development roots; S1 eight-decision autonomous check against H12 (≥90% aggregate, ≥85% per family, deterministic and stochastic); housing/cloth/farming share against teacher on matched states.
- **Activity fidelity within staffed roles** (from the 24 Oct arm3-v2 diagnosis, `model9-data/diag-arm3v2/findings.md`): on development roots, compare the actor's per-role activity shares with the teacher's on the same states, in particular Craftsperson time on housing and housing material (teacher about 32% each, arm3-v2 about 3%). Report them per role, weighted by staffing. The same diagnosis found that specialty-level shares and integer rounding were not the cause.
- **Critics:** normalised MAE ≤ 0.20; |bias| ≤ 0.10; ranking ≥ 70%; zero-cost heads ≤ 0.05; per head and per stratum.

## 7. Budget (fill from measured throughput)
| Phase | Work | VM estimate |
|---|---|---|
| BC collection | 80 worlds × 25 decisions | TBD from G1 throughput |
| Critic anchors | 100 worlds × roots × 16–64 streams | TBD |
| BC fitting | 50 epochs on the collected rows | TBD |
| Critic fitting | 100 epochs | TBD |

## 8. Amendment draft for round r5 (26 Oct) — privileged critics, advantage head, held-out Q-π arm
Source: `intervention-brief-2026-10-26.md` (user's intervention brief) plus the user's decisions in chat. The G0 run on the VM (stages D/E) stays frozen as the feedforward control.

**Actor comparator.** Actor gates (S1, G4) are judged against H12, the converted heuristic through the action-12 interface, as brief §5.2 states: ≥ 0.95 of H12 in aggregate, ≥ 0.90 per family/world, in both modes. H is reported separately as the interface-quality check; H12/H was 0.965 on 25 Oct.

**Preregistered numbers.** As in the brief's table:
- privileged row caps: fields 16, herds 8, unlocks 1, economy 1, each with presence and overflow;
- value support [-0.5, 2.5] × 255;
- randomised tail with prefix 25, 16 draws, rho 0.95, cap 200, or else an explicit age feature in all arms;
- critic budget as critic3 (4 epochs), with a second arm at 12 epochs;
- QoL acceptance: nMAE ≤ 0.20, |bias| ≤ 0.10, ranking ≥ 0.70, other heads regressing ≤ 0.02;
- advantage head: 128-d action embedding and a two-layer MLP;
- pair exclusion at 2 SE, same horizon and stream only;
- ranking acceptance ≥ 0.70, with false improvement ≤ 0.15 at a 0.01 margin;
- pruning: top 4 plus H12, with full evaluation on 20% of roots;
- seeds: at most 3 locally, 5 paired on the VM, 3000–3059 sealed.

**Held-out Q-π arm (user decision, 26 Oct).** The plumbing is built with intervention B, flag off. The actor's reparameterised sample feeds B's action encoder, so Q(C, a ~ π) can later drive many actor updates on stored roots.
- **Reserved worlds:** 10% of the fitting bank, held out entirely. These are fitting worlds **90154–90159** (6 of 90100–90159; none of them are in the G4 DAgger set 90100–90129). Nothing from these worlds or their descendants may train Q, A, V or the actor. Their search rows from the G0 run are kept only as a test set.
- **Admission order before any critic-driven actor update:**
  1. QoL V passes G2C with privileged inputs.
  2. B passes held-out ranking (≥ 0.70, false improvement ≤ 0.15).
  3. Q is fitted and calibrated under the frozen actor (G5-style).
  4. A bounded update canary runs with a KL cap and parameter+Adam rollback, judged by native re-evaluation on the reserved worlds and by closed-loop welfare against H12.
- **Failure checks:** Q exploitation/overestimation, policy drift away from the Q fit, and personality leakage.
- **Evidence precondition (user, 26 Oct) — the escalation is earned, not scheduled.** The Q-π arm may be considered only after both hold:
  1. **A well-conditioned critic.** The QoL V head passes G2C on the calibration worlds (nMAE ≤ 0.20, |bias| ≤ 0.10, ranking ≥ 0.70), with the other heads not regressing. Without this, nothing downstream is interpretable.
  2. **Measured inadequate QoL disparity in the existing advantage signals,** on the held-out roots. Proposed tests, to be fixed before the r5 collection:
     - the QoL component of the state-value (GAE) advantage separates natively re-evaluated candidate pairs beyond 2 SE at below 0.70 pairwise accuracy;
     - and/or B's A_QoL ranks those pairs at below 0.70 while its population-years and cost heads pass;
     - and/or, on at least a preregistered share of roots, the spread of predicted QoL advantage across candidates is indistinguishable from the 2-SE noise floor, where native returns do differ.

  If the existing signals already discriminate QoL adequately, the Q-π arm stays closed: the evidence would show no need for it. These thresholds are a proposal for the user's review.

Status: draft. It needs the user's review before any r5 collection.

## 9. Amendment r5-b (26 Oct, user-approved): the action-conditioned critic is foundational
Root cause, as the user stated it: the critic's output heads were never reconditioned when welfare was refactored out of the reward. QoL is bounded to [0, 1] and nearly constant across sister camps, so a cross-village ranking test and a symlog two-hot value head were the wrong instruments for it.

1. **The action-conditioned critic is core, not held out.** The critic is Q(state, privileged state, action) = V + A. The action is the actor's own output head, through intervention B's fixed-layout action encoder. A is trained on same-village alternatives restored from an identical root (paired differences beyond 2 SE, centred over the candidate set), so a village competes only with itself. **QoL acceptance is defined by A:** the critic must tell the actor whether its intended action makes this village better or worse than the alternatives. A state value alone cannot, because the next state depends on the action.
2. **Actor updates through Q stay gated.** Before any actor gradient flows through Q, these must pass on the reserved worlds 90154–90159:
   - same-village pair ranking ≥ 0.70 on pairs beyond 2 SE;
   - false improvement ≤ 0.15 at a 0.01 margin;
   - the all-idle and delayed-benefit controls.

   This replaces the earlier "held-out Q-π arm" framing. The plumbing is now the main critic, and the update path is what stays closed.
3. **QoL acceptance criteria, in priority order:**
   - (a) same-village action ranking via A, the primary criterion;
   - (b) calibration on the bounded head (nMAE ≤ 0.20, |bias| ≤ 0.10);
   - (c) temporal consistency (TD residuals) along a village's own trajectory.

   Cross-village ranking remains a criterion for population-years only.
4. **Bounded QoL value head.** It predicts discounted-mean QoL in [0, 1], rescaled by the known discount weight, and replaces the symlog two-hot for QoL only; population-years keeps its distributional head. The default output is logistic. A selectable fallback (SiLU pre-activation, clamped to [0, 1] at decode) stays available if the logistic path saturates or its gradients vanish, along with a saturation monitor: the share of pre-activations with |z| > 6, reported every fit.
5. **Sibling-competition audit (reward credit, separate from the critic).** Check sibling outcomes after fission and resource transfers between sister camps, to see whether a camp-centred reward with short-lived lineage sharing teaches a policy to win contests against its own daughters.
6. **Separate pretraining first; multi-step querying later (user, 26 Oct).** The actor and the critic are pretrained separately at first, with no cross-network values or gradients, as in brief G2A/G2C.
   - **Actor:** BC on heuristic (H12) labels plus search labels, with DAgger rounds on learner-visited states.
   - **Critic (V and A):** trained on the same mix of states. The two sources supply different targets:
     - **V targets:** teacher (rho) continuations from both teacher-trajectory and DAgger learner-visited roots. Today's DAgger shards are actor-only, because their rewards come from the learner branch, so learner-visited roots need a teacher continuation, with the r5 tail method, before they can give V targets.
     - **A pairs:** search run at both H12 roots and actor-visited roots (`labor_search --source actor`).
   - **Later, once Q passes its gates:** the actor may query a *pretrained* Q and take multiple optimisation steps against it on stored roots. That stays the intended use, opened only through the item-2 gates and a KL-capped, rollback-protected canary. A change to the actor invalidates Q's binding, so Q is refitted before the next batch of updates.

## 10. r5 FROZEN (user, 26 Oct 03:15Z)
r5 is approved and frozen: §8, §9 and these decisions.
- **Critic targets:** the randomised tail (prefix 25, 16 draws, ρ 0.95, cap 200) for every critic.
- **QoL target:** TD(λ) bootstrapped through the bounded head, λ = 0.8, replacing the 25-decision Monte Carlo return. Population-years and the costs keep the randomised-tail Monte Carlo return.
- **DAgger teacher continuations:** branch ends only, on a keyed 25% of roots.
- **Code:** learner `feature/r5-integration` bf39c7c; simulator `feature/r5-privileged` 8bd480b.
- **World partition:**
  - fitting: 90100–90153 and 90160–90165 (60 worlds);
  - critic calibration: 90180–90199;
  - reserved (held-out tests only): 90154–90159;
  - development: 2000–2059;
  - sealed: 3000–3059.
- **Normalizer:** the actor normalizer is refitted on r5 fitting worlds only, because the action-12 context fix changes the core fields; the privileged normalizer v2 is fitted on fitting worlds only.
- **Actor schedule:** the brief's learning-rate schedule (peak 3e-5, 10% warm-up, cosine to 10%), Adam carried across rounds, and a fixed 24-world development panel for every evaluation.

## 11. Acceptance thresholds — user decisions (r5 thresholds doc)
- **§2.1 non-finite rollback:** approved as proposed. Abort after 3 consecutive failures; a fit fails after more than 5 rollbacks. Escalation: rollbacks or aborts in more than one seed of an arm halt admission for that arm and go to the user.
- **§2.2 gradient dominance:** the window is **400 optimizer steps** (16 log points), not 200. Other parameters as proposed. Commit 5th on r5-integration after 8e5c9b8; acceptance fits use this or a later commit.
- **§2.3 QoL saturation:** approved with a relaxation bound.
  - Low-side saturation where the bounded target is below 0.05 × W counts as expected and does not trigger the alarm.
  - The first 5 decisions per trajectory alarm only above 20%; all other rows alarm above 5%.
- **VM persistence (user, 26 Oct):** the r5 `@reboot` crontab line (`~/r5/loop.sh`) is approved for this run. Remove it when r5 finishes.
- **§2.4 TD drift:** a trajectory rule, not an absolute bound.
  - Ignore the first 2 refreshes (warm-up).
  - Target change and TD error must then decline: negative log-slope, and a final value at most 0.5× the post-warm-up peak.
  - Divergence is 3 consecutive increases of more than 10%.
  - Keep the hard [0, W] bound.
- **§2.5 population decoded tail:** approved and gated. ESCALATE any trigger to the user: it is a symptom of a control problem in the shared trunk's population-years output.
- **§2.6 A overfitting:** approved.
  - Standing remedy on a trigger: more scenario roots, plus candidates from a broader variety of heuristics.
  - Escalate to the user ONLY on regression traceable to insufficient trajectory knowledge (history-dependent outcomes A cannot separate). That is the evidence for a GRU critic, which the current brief does not authorise.
- **Critic fit budget:** 4 epochs (passes over the data) stays. The user's "25" referred to game decisions, not epochs; clarified 26 Oct.
- **Collection horizon (user, 26 Oct):** the recorded teacher window goes from 25 to **50 decisions**, plus the randomised tail beyond it. 100 decisions is pending a memory measurement from the 50-decision run.
- **Curated Neolithic scenarios (user, 26 Oct):** coverage-driven gap filling, not a parallel bank.
  - Audit family coverage in the 50-decision banks first.
  - Then harvest roots only for families below the support minimum: ≥ 32 fitting roots across ≥ 4 worlds, and ≥ 16 calibration roots across ≥ 4 worlds.
  - Families: head first-use (farming, herding, fishing, mixed), small or large camp, under-housed, under-clothed, fission daughter, trade-eligible.
  - Fitting worlds only for training; calibration worlds form the per-family evaluation set.
- **§2.7 privileged attention:** approved as an advisory check (0.10 × the uniform share).
- **§3.2 generalisation gaps:** approved at 0.10. A gap above it means insufficient critic training data; the remedy is wider data (worlds, horizon, scenarios), never a looser gate.
- **§3.1 TD residual:** approved as a rolling k = 5-decision n-step residual along each trajectory. Aggregate: |bias| ≤ 0.05 and mean |residual| ≤ 0.20. Per 10-decision position group: |bias| ≤ 0.10 (groups with fewer than 100 windows are reported, not gated).
- **§3.3 seed rule:** mean-with-floor. The seed mean passes, the worst seed is within 0.02, and every seed passes stability. The driver sets SEED_RULE=mean-with-floor.
- **§3.4 determinism:** approved at 1e-6 relative.
- **Priority (user, 26 Oct):** the r5 critic Q(s, a) = V + A passing its gate comes first. Search with full payloads (A's data) moves onto the critical path. Actor pretraining waits for the critic receipt.
- **Search scale:** 16 roots per fitting world; at least 32 held-out roots per reserved world (90154–90159).
- **Actor evaluation panel:** 24 development worlds × 50 decisions, treated as a scale-out test; comparisons with G0's 8×8 panel are indicative only.
- **Approved:** QoL calibration on all five strata; HOLD when more than 5% of TD windows are unavailable.
- **Pending the user's go-ahead:** a GPU spot VM for the fitting stages.
- **T2 plan template (user, 26 Oct):** `housing-grow` becomes `shelter-and-clothe`: build housing and clothing. Death rate depends on clothing and shelter, and food-capped settlements stall without them.
- **Trajectory-diagnostic thresholds:** tentative. They are recalibrated from the first real pretrain diagnostics.
- **CUDA determinism:** approved at 1e-5 relative; CPU stays at 1e-6.
- **GPU VM network:** option (a), with no external IP, Private Google Access and an offline wheelhouse through GCS.
- **Quota requests submitted by the user:** GPUS_ALL_REGIONS = 1 and CPUS_ALL_REGIONS = 80.
- **Scenario weighting (proposed, awaiting the user's confirmation):** no new collection. Up-weight family rows in the existing banks: first-active-use ×10, legal window ×5, fission daughter ×2, under-housed/under-clothed ×3, renormalised per world.
- **Scenario weights APPROVED (user, 26 Oct), with conditions:**
  - the first-active-use (and legal-window) weights decay toward ×1 over training, cosine by default, and continue across DAgger and refit rounds without restarting;
  - a never-up-weighted `food-trade` family is reported per receipt;
  - an advisory flag fires if food-trade calibration or ranking is more than 0.05 worse than all rows, so up-weighting doesn't silently penalise food-trade villages.
- **Registered contingency (user proposal, 26 Oct): an exploration pressure on labor allocations during PPO.**
  - **Trigger:** newly exposed heads are under-used, with adoption in unlock states below the teacher's or search's beyond seed spread, or the weight decay produces second- or third-order effects.
  - **Form (co-architect refinement):** a per-head-family entropy floor on the policy's allocation distribution, the latent dispersion for specialty and activity. It is not applied to attention weights. An adaptive temperature drives each family toward a target entropy, applied mainly in head-unlock windows and decaying as adoption matches the teacher's.
  - **Guards:** no allocation quotas; sampled-policy outcome checks (brief §17.3); per-head KL within the PPO budget.
  - **Phase:** PPO only, not BC pretraining. It stays dormant unless the trigger evidence appears.

- 2026-10-26 14:45Z (reviewer): scenario work rebased into feature/r5-integration at 51c139a (70e938d..51c139a fast-forward).
  - Weighting math reviewed: the per-(source, world) closed-form rescale B/(B+s(M-B)) is exact, max-over-families, cosine decay 1+(w-1)s(t), offset continued via checkpoint, decay=none is bitwise-identical. Accepted.
  - Noted: decay_epochs defaults to the first fit's epoch count, so weights reach x1 by the end of the first fit and later rounds run at x1. Consistent with "decay with time"; the user can lengthen it.
  - Open for the user, NOT changed: under-clothed is about 23% of rows even with hysteresis, so x3 gives about 39% of the weight at s=1; up-weighted rows total about 89% at s=1 (one world). Reviewer recommendation: under-clothed x1.5-2 (it is a prevalent state, not a rare window). No fit uses the weights before the morning sync.
  - Food-trade guard is unmeasured: stored rows may have the imported-food column zeroed by the normalizer. Operator to check on FIT50 with the r5 actor normalizer; if it is zeroed, that is also a dead actor feature, and the bank needs a raw imported-food field.
  - origin/feature/r5-scenarios (86a0633) is stale pre-rebase; left in place, deletion is the user's call.
- 2026-10-26 14:20Z: auto-restart schedule extended to g0-autostart5 (ends 2026-10-28T12:00Z).
- 2026-10-26 ~15:10Z (operator report + reviewer):
  - The wheelhouse is built and uploaded (2.9 GB, 43 files). The resolver chose torch 2.14.1+cu130 (CUDA 13.0) despite CUDA_TAG=cu128. Accepted: the 580 driver is CUDA-13 capable, and the parity tests are the check.
  - Collection is far slower than estimated.
    - The tail averages ~50 decisions at ~87 s each.
    - The window takes 1.1-2.4 min per decision.
    - Projection: 25-30 h for all 9 batches, finishing ~27 Oct 14:00-19:00Z.
  - Memory: peak 6.0 GB per process at ~decision 100 (tail), so JOBS stays at 8. A 100-decision bank fits only ~16 processes on this VM.
  - Disk: 46 GB free against a projected 35-45 GB of shards. I resized the boot disk to 200 GB (online); the operator grows the filesystem. G0 data is not deleted.
  - Evaluating using the idle GPU VM as an extra collector for 1-2 batches (needs a self-contained bridge publish). Feasibility first; nothing is run yet.
- 2026-10-26 ~17:45Z (reviewer): CPU-vs-CUDA parity redesign at a204cf5, fast-forwarded into feature/r5-integration.
  - The multi-step per-tensor checks were ill-posed: gauge parameters (attention key bias; the A last bias under per-root centring) plus Adam sign-steps on roundoff gradients.
  - Replacements: an output-space check ≤ 1e-3 and a teleported mid-step check ≤ 1e-4 (loss, global grad). Parameter drift is reported, not gated.
  - L4 result: 24/24 pass, CUDA repeats bitwise.
  - TORCH_DISABLE_NATIVE_JIT=1 is required for CUDA (configure refuses otherwise).
  - The approved CUDA determinism gate of 1e-5 is unchanged; the remedy if it fails is pinning SDPA MATH, never loosening.
  - GPU accepted for the r5 critic fits.
- 2026-10-26 ~18:10Z, USER: on-demand allowed if preemptions continue to substantively impede pretrain progress, keeping the same rules (VMs stopped when idle, max-run-duration caps, no idle quota use).
  - Reviewer trigger, applied only AFTER collector resume is deployed: switch a VM to on-demand when preemption downtime (restart gap plus redone work) exceeds 25% of its wall time over a rolling 6 h, or when 2 consecutive batches fail to finish.
  - On-demand VMs keep max-run-duration, are stopped the moment their queue is empty, and revert to Spot when the stage ends.
- 2026-10-26 ~22:00Z (reviewer): simulator [NonSerialized] leak (EconomicLaborPlan.ActorActivityAuthority/Actor*Workers not cleared at the boundary).
  - Measured on 90100 (16 players, 16 decisions):
    - 0/771 shadow labels changed;
    - 0/830 native teacher records changed;
    - rows, observations and rewards bitwise equal after a snapshot-only branch;
    - only the DecisionTrace move scores (±0.05) and the world fingerprint differ.
  - Search, DAgger and scenario forks are symmetric (the root and candidates share the reload).
  - Decision: TOLERATE for r5. Collector resume keeps the transient restore plus the digest check, for fingerprint-exact continuation.
  - The simulator fix (clear it at the boundary per its comment) is the USER's call. It changes simulator identity and replay fingerprints, not rows.
  - Residual risk: a near-tie heuristic move flip (inertia 0.18); none seen in 830 decisions.
- 2026-10-27 ~03:30Z, USER decisions:
  - (1) Under-clothed family weight reduced from x3 to x2 (the user agreed to x1.5-2; reviewer picks x2). Pass it as an explicit --family-weights override in the critic and BC fits; the other defaults are unchanged.
  - (2) Queued for r6: refit the actor normalizer with a wider input range. The imported-food column saturates at 1.72 food (std floor 0.1, ±10σ clip). Audit every floored/clipped trade column. This needs recollection.
  - (4) bootstrap.sh in GCS replaced with the version that exports TORCH_DISABLE_NATIVE_JIT=1.
- 2026-10-27 ~03:45Z, USER:
  - Simulator transient-state leak: patch in r6 (alongside the normalizer refit); leave it for r5.
  - Provisioning: on-demand through collection and the critic fits, then back to Spot; keep the 8-hour guardrail on.
  - Reviewer note on reusing the critic in r6: one checkpoint carries ONE observation normalizer shared by the actor and all critics, and stored rows are normalized. The privileged tokens are raw with their own normalizer and have meals-per-turn and trade-partners but no imported-food amount. So the critic sees the same clipped column.
  - Proposed r6 path: a targeted normalizer change (widen only the floored/clipped columns, freeze every other column's stats) plus a warm-start critic fine-tune and a gate re-run. That avoids a full critic retrain. Alternative: separate actor and critic normalizers (code change; the critic stays blind to food-import size).
  - USER APPROVED (27 Oct ~03:50Z): the r6 path is the targeted normalizer change (widen only the floored/clipped trade columns; freeze every other column's stats at r5) plus a warm-start r5 critic fine-tune on r6 data and a re-run of the same critic gate.
- 2026-10-27 ~04:30Z, USER: add early-detection guards before the r5 scale-out.
  - (1) Read-only harness audit (Sonnet ad5c2d7169ef85d47): which checks act live and which only at the end; convert end-only checks into early stops or operator-watched JSONL triggers.
  - (2) Pilot critic fit on partial data, NOT gate evidence: fit on fit8-b01+b03, validate on fit8-b04 (overfit and calibration); provisional privileged normalizer; 1 seed, privileged arm; CUDA; ≥ 4 TD refreshes. The GPU VM is being recreated on-demand (Sonnet a4b59f4231c0970d2); code tarball b6a1290 (sha 960fe799…).
  - (3) Scale-out runs seeds sequentially with an early stop: if the first privileged-arm seed fails stability, stop the arm and diagnose.
- 2026-10-27 ~06:30Z (reviewer): early-stop guards d9555a3 reviewed and fast-forwarded into feature/r5-integration.
  - Live StabilityAbort on would-FAIL dominance, TD divergence and >5 rollbacks (same check ids and thresholds).
  - Population-tail live alarm (advisory).
  - Per-epoch V validation on CAL_DATA with a validation_divergence advisory. Training is bitwise unchanged, tested.
  - Seed-0 gate on the candidate arm.
  - Accepted deviation: validation runs in train mode (dropout 0, guarded) to avoid the fused eval fast path.
  - Tarball r5-int-rl-d9555a3 (sha 45f776f6…). The pilot now uses --stability-early-stop --validation-dataset fit8-b04.
- 2026-10-27 ~12:40Z (reviewer): b03 failure root cause is a LATENT label-conversion bug, not a resume regression.
  - World 90119 turn 120, camp 327, Builder role: housing and housing-material at 0.5000000054 each, sum 1 + 1.09e-8.
  - The row passes LABEL_TOLERANCE (1e-6) but fails SIMPLEX_TOLERANCE (1e-8). It is identical on c9fc899 and with transient restore on or off.
  - Fix e6f8b67 (fast-forwarded into integration): quarantine via the existing over-budget path with the reason activity-simplex-tolerance-overrun:role<r>; no repair.
  - Byte-identical shards elsewhere. Published data is unaffected: the case always crashed, never silently.
  - Recovery: rerun b03 in full as fit8-b03r under e6f8b67; keep the old dir as evidence.
  - New launches move to e6f8b67. Running b6a1290 batches stay pinned to their own commit for resume.
- 2026-10-27 ~17:00Z, USER APPROVED (pilot A diagnosis):
  - (1) Replace the flat QoL [0, W] target check with a per-row admissible bound U_i (same recursion; head max for V; max(tail, head_max) on the last row). It still flags reward ∉ [0, dt], wrong γ and decode out of range. Same thresholds.
  - (2) Widen the bounded QoL head from (0, W) to (0, 2W) via a versioned qol_head_scale (absent = 1.0). QoL metrics and gates stay in units of W.
  - Plus a non-gating per-world tail-factor diagnostic next to the QoL gates.
  - Reviewer: the population-tail FAIL at step 0 is an untrained-head artifact. Targets fit the support (symlog max 1.96 < 2.5).
  - QUEUED FOR r6: store per-step continuation rewards and use per-camp independent stop schedules (exact truncated tail plus V bootstrap: bounded, no world bias). It rides the r6 recollection.
  - Implementer: Opus, branch feature/r5-qol-bound.
- 2026-10-28 ~01:45Z, USER APPROVED (after pilot E):
  - (1) A separate QoL encoder in the task critic (architecture change; pilot first on b01 → b04).
  - (2) The QoL calibration gate measures against the tail-robust target (n-step bootstrapped, as in the TD-residual report), with the QoL reference scale remeasured on r5's own returns.
  - (3) The TD trajectory rule is replaced by a well-posed criterion: PASS if declining (slope ≤ 0 and final ≤ 0.5 × post-warm-up peak) OR stationary at a floor (last K = 6 refreshes with range ≤ 0.5 × median, median ≤ a declared floor in W). Divergence = 3 consecutive >10% rises ending above 1.5 × the stationary median. The error series is the primary gated series.
  - (4) Tail redesign moved INTO r5: store per-step continuation rewards per camp (all heads), with independent per-camp stop draws over the shared world continuation. This needs recollection of every batch collected under the old format.
  - USER (01:50Z): HOLD item 4 (tail redesign) until 1-3 are done. Old-format collection continues meanwhile (keeps the option open if 4 is later dropped).
- 2026-10-28 ~04:00Z (reviewer): gate v2 merged into integration (55774fc + bad7afc).
  - QoL nMAE gate vs nstep-k5-bootstrap-v1.
  - QoL |bias| gate vs mc-tailcut-0.5-v1: world-balanced MC bias on rows with tail weight ≤ 0.5, ≥ 200 rows and ≥ 3 worlds per stratum, else HOLD. Chosen because an n-step bias sees only ~0.22× a uniform offset.
  - r5 QoL reference-scale artifact (MC per-row std on fitting batches).
  - TD rule v2 (floor-or-declining, error primary, drift advisory). Pilots E1/E5 re-judge PASS.
  - Known limitation: the tail-cut bias is still sensitive to the calibration set's MEAN tail factor (robust only when world factors balance). The receipt should show the calibration-set mean schedule factor. A lower cut (0.2) would tighten it but thins the late stratum. To raise with the user.
  - USER (04:10Z): keep the 0.5 cut; show the calibration-set mean tail factor and attribute the bias to tail noise vs the critic. Amend requested: a non-gating predicted tail-induced bias, the residual critic bias, and advisory:qol-bias-tail-attribution when |pred| > 0.05 or the verdict would flip.
  - USER approved the Fable audit escalation gate (28 Oct ~08:05Z); auto-launch if any F1/F2 trigger fires.
- 2026-10-28 ~10:30Z, USER SIGNED OFF (gate philosophy): the critic must answer "given state and action, is the action / the next state better for the elder's reward?". Relative accuracy is primary; absolute calibration is secondary.
  - Planned restructure:
    - PRIMARY: (1) same-state A ranking ≥ 0.70 on reserved worlds; (2) next-state improvement: one-step/n-step advantage sign accuracy vs realised outcome, plus within-trajectory V ordering; (3) TD-residual self-consistency; (4) stability.
    - DIAGNOSTIC: absolute nMAE/bias (both heads).
    - GUARDS: a loose absolute-bias sanity bound; consistent QoL/population head scale tested via combined-advantage ranking.
  - SEQUENCING (user): first get the saturation audit and decision-trace results (they may show immediate problems), THEN implement the relative-metric diagnostics and write the gate-restructure doc for approval.
- 2026-10-28 ~16:30Z, USER DECISIONS:
  - (1) Gate restructure APPROVED. Primary: within-trajectory ordering at lag ≥ 5 with a 0.02W margin ≥ 0.70; all-pairs ordering; TD(0) residual self-consistency; stability; same-state A ranking on reserved worlds ≥ 0.70. Diagnostic: n-step and MC nMAE, printed with the constant-predictor reference.
  - (2) Tail redesign moved back to r6.
  - (3) FIX the simulator's heuristic clothing rule (2-turn cycle) and the false-terminal rule (Terminated = Population ≤ 0 while hunters are away, ElderModel9.cs:1075).
  - Reviewer tweaks proceeding: QoL head bias init at logit(mean p*) with step-0 TD targets from MC; per-world tail normalisation in the loader; check_support allclose 1e-6; shared encoder w30 as baseline; the campaign default reverted to shared.
  - Open question raised with the user: which data the simulator fixes apply to (r5 recollection vs PPO/r6 onward).
  - USER (16:45Z): the simulator fixes apply from PPO/G5 and r6 onward; r5's critic trains on existing data with false-terminal rows masked loader-side (item 6 added to feature/r5-relative-gates).
  - USER (18:30Z): combined elder-reward variant GATES; QoL and population variants are diagnostics. TD(0) MAE ≤ 0.12 S and |bias| ≤ 0.02 S APPROVED. All-pairs ≥ 0.65: user asked what it is measured against (answered: agreement with the realised elder reward-to-go under the teacher's policy).
  - USER (18:45Z): all-pairs ≥ 0.65 APPROVED (combined variant). Intent: PPO needs advantages with the right direction and order of magnitude. Asked for a non-gating advantage-magnitude diagnostic (sign, slope, std ratio, k = 1, 5) with proposed bounds.
  - USER (19:00Z): SIGN AGREEMENT is the key advantage metric (likely to gate). Measure RELATIVE per-head advantage sizes, since PopArt dynamic normalisation is expected during PPO. Sent to the implementer.
