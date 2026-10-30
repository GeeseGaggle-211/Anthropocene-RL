# r5 critic–actor fit and normalization backlog

Started 29 Oct 2026. This lists what has to be added for the critic and actor to fit together in PPO. Status key:
- **Approved:** the user approved it.
- **Proposed:** it needs the user's approval.
- **In progress:** it's being built.
- **Done:** it's merged.

## Current wiring (read 29 Oct)

The combined advantage is built in `labor_ppo.py:402` `combine_labor_advantages`:

    A = w_q · A_qol / s_q + w_p · A_pop / s_p − Σ_c m_c · A_c / s_c

- **Task scales** `s_q`, `s_p` come from `contracts.py:36-40` (Round 3, old simulator, Small world):
  - QoL 43.152; r5 measured 55.67, so stale by 1.29×.
  - Population 85,133.98; r5 measured 135,589, so stale by 1.59×.
  - Net effect: population is over-weighted ~1.23×.
- **Cost scales** `s_c` are 1.0 (`labor_model.py:73`), so costs enter raw. Cost multipliers `m_c` default to 1.
- **The same task scales also define the critic's target units and value supports** (`labor_critic_support.py:127`, `labor_bc.py:3413`). Do not change them while r5 checkpoints exist; normalize on the PPO side instead.
- **PPO feeds A into the clipped surrogate raw,** with no per-batch standardization (`labor_ppo.py` attempt_policy_terms, ~559).
- **The panel gate metric is welfare in person-QoL-years** (≈ Σ population × QoL; `labor_pretrain_eval.py:2604-2628`) plus person-years against H12. PPO optimizes a linear combination of the heads, so the two agree only to first order.

## Per-head inventory

| Head | Role | Training | Normalization | Issues | Status |
|---|---|---|---|---|---|
| Critic: qol_years (logistic, TD-λ 0.8) | V for QoL | BCE soft target; QoL weight 30 initial | Target units: s_q (frozen) | The weakest head (ordering ~0.70–0.73); stalls under population dominance at 60 worlds | GradNorm in progress |
| Critic: population_years (two-hot symlog) | V for population | CE | s_p (frozen); fixed support | Dominates the shared encoder at 60 worlds; widespread top-bin mass (population_tail, advisory in r5) | GradNorm in progress |
| Critic: feasibility (cost) | Cost V | Separate cost encoder | s_c = 1 | Constant-zero in practice: 0 rejected attempts in r5 data. Meaningful only after G5/PPO data | Proposed: normalize in PPO |
| Critic: settlement_guard (cost) | Cost V | Separate cost encoder | s_c = 1 | Already ~as large as QoL in the combined advantage (pred 0.41 vs QoL 0.38 of the spread, b04 E5) and over-predicted (realised 0.24) | Proposed: normalize and set multiplier |
| Critic: A head (action-conditioned) | Same-state ranking, gate (d) | Needs search payloads | Per-head reference scales | Waiting for search (running) | Pending data |
| Actor: specialty / activity / movement / fission / trade-gate | Policy | BC pilot done (spec L1 0.114, act L1 0.082, move 0.98, fission/trade 1.0; no fit/held-out gap) | Actor normalizer fcc769c3 | The panel against H12 isn't run yet | Panel pending (after search) |

## Bolt-ons for the critic–actor fit (PPO stage)

1. **Per-head PopArt (or running-std) advantage normalization in PPO, for both task and cost heads.**
   - It replaces the fixed `s_q`, `s_p` and `s_c=1` inside `combine_labor_advantages` only.
   - Critic target units stay frozen.
   - Seed the statistics from r5 measurements: QoL A_MC std ~28–31, population ~2.3–2.9e4, settlement_guard Â std ~0.10. The full per-stratum moments are in the advantage-magnitude reports under `heads.<h>.popart`.
   - If the critic is refit online (G5), use true PopArt, which rescales the value heads' output layer to preserve outputs.
   - Status: proposed. The user signalled it's needed (28–29 Oct).
2. **Welfare-linearized per-row weights.** Set w_q ∝ the camp's population and w_p ∝ its QoL, per row, via the existing `combine_personal_task_advantages(row_weights)`. The training objective then tracks the person-QoL-years gate to first order without new heads. Status: proposed.
3. **Deliberate cost multipliers** once the costs are normalized, so feasibility and settlement-guard act as constraints rather than co-objectives. Status: proposed; the user decides.
4. **Per-batch advantage standardization (optional)** on top of PopArt, as a safety net for the raw-A surrogate. Status: proposed.
5. **Critic GradNorm on the task heads plus a separate clip per encoder,** with `task_progress_stall` gating and dominance advisory under GradNorm (`feature/r5-gradnorm`). This is the user's backup and escalation gate, which fired on 29 Oct (seed 0 tripped at weight 30 and at weight 50). Status: in progress, then a 1-seed 60-world pilot.
6. **Teacher V vs actor V.** The G5 refit on actor trajectories, plus DAgger continuations, so advantages reflect the actor's policy. The teacher's 2-turn clothing cycle is fixed in the simulator (`feature/r5-sim-fixes` ddafab8) for PPO, G5 and r6. Status: approved (sequencing).
7. **Dormant-camp bridge handling,** needed before PPO runs on the fixed simulator. The collector must accept dormant successor rows (`labor_bridge.py:1924`, `:2069`) and discount gaps by true elapsed years. Dormant rows (population 0, not terminated) must NOT count toward settlement_guard (`labor_training.py:294` would charge 1 per dormant turn). Status: required with the simulator fixes.
8. **A head and gate (d)** after the search payloads land. Then actor-root search from the pilot actor once it passes fidelity. Status: approved (sequencing).

## Done (for the record)

- Relative-first gates (combined variant gates).
- Tail world-factor normalization; false-terminal mask.
- QoL bias init; step-0 MC targets.
- check_support tolerance.
- Population tail advisory; control-arm reference.
- Search root-level resume (selftest passed 29 Oct).
- Open-file fix for mmap loaders.
- Moved to r6: the tail redesign.

## Notes (29 Oct)

- **PopArt applies on the PPO side only.** The critic's QoL head is a bounded logistic and its population head a fixed-support categorical, so neither can be rescaled in the PopArt way. Use PPO-side per-head statistics.
- **LayerNorm and saturation are not the QoL bottleneck** (saturation-audit.md: no hidden or head saturation; float32 is fine). QoL is limited by a weak learnable signal (~0.003 nats of BCE excess) and by gradient competition. The levers are GradNorm (in progress), more worlds, and the tail normalization already in place.
- **Feasibility cost** (`ElderActorLaborLedger.cs:243`) is 0, or 1 + Σ severity of avoidable, actor-visible violations, up to 319 per attempt. It is always 0 in r5 teacher data, and it would swamp raw advantages in PPO without normalization or a capped multiplier.

## Candidate: inter-village inputs for the critic (user idea, 30 Oct)

- **Hypothesis:** the critic sees one camp, while fission, emigration and immigration depend on other settlements, so QoL looks noisy to it. Population-years should suffer too but is masked by its scale and growth rate.
- **Diagnostic (approved; runs after pilot 4):** on the local b01 copy, split critic QoL and population error and ordering by:
  - a fission/migration event within 5 decisions;
  - the teacher clothing-cycle phase.
- **If the error concentrates on events:**
  - Step 1 (tweak): privileged k-nearest known-settlement features: their QoL, population and distance, plus this camp's fission and immigration utilities.
  - Step 2 (architecture, user decision): a centralized critic attending over all villages' input and privileged tokens in the world (CTDE; the actor stays decentralized).
- **Caveats:**
  - More inputs per row do not add rows; the effective sample size stays the number of worlds.
  - World-level inputs make world fingerprinting easier: privileged unlocks already identify a world 58% of the time.
  - Cost scales with village count, which grows with fission.
