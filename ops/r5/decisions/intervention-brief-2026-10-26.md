# Intervention brief: privileged critic inputs and advantage head

Oct 26, 2026 · @Priya Loganathar. Copied verbatim from the user's message so implementers can read it.

These are instructions for the session driving the G0 program. There are two changes only: critic-only privileged state, and a scalar action-conditioned advantage head used as a search ranker. Everything else in the proposal review stays out of scope here.

## Scope, timing and ordering

Both changes land in the next registered round, not in the G0 run now on the VM. Let stages D and E finish and upload; the frozen run is the feedforward control every later comparison needs. Do not restart the running G3 search to add payloads; its output is still valid as labels and as a ranking test set for intervention B.

Order of work, each item its own commit with a test:

1. Hygiene that both interventions depend on: value support stays at [-0.5, 2.5] by 255; replace the zero tail at turn 25 with the randomized-tail estimator already used by the anchors, or add an explicit episode-age feature to every critic input; make pretrain-critic reject search shards; rename the years field that is labelled days.
2. Simulator fixes that both interventions read through: the action-12 per-village native context that never updates (ElderModel9.cs near line 989), and the section 18 ledger repairs (Herder staffing inferred as zero, migration person-equivalents mixed into food, missing herd mutation events). Privileged features derived from those ledgers are wrong until this lands.
3. Intervention A (privileged state), simulator side then learner side, then one re-collection of fit and calibration data.
4. Intervention B (advantage head), trained on the re-collected search export, validated on held-out roots, then switched on as a ranker inside search.

What this brief does not authorize: any GRU or hidden state, any actor update driven by the critic, any change to the actor's inputs, any change to G2C's V criteria, any change to the sealed seeds 3000 to 3059, and any touch of a live VM run. A new reviewed manifest is required because both changes are material to the critic architecture and data contract; the review tab's section on brief fit lists the gate fields that change.

## Intervention A: privileged state into the critics

Give the three critic encoders (task, feasibility, settlement guard) a versioned critic-only view of the hidden economic state, and keep the actor observation-only. The ten floats the simulator already emits become family 0 of that view; the brief's typed families are added around them.

### Simulator (branch from feature/g1-fixes a645431)

1. Add a PrivilegedSchemaVersion to the model9 contract next to the existing feature-name list (ElderModel9.cs lines 74 to 76) and write it into every emitted row and into privileged.npy metadata. The current vector has no version.
2. Keep the ten existing floats unchanged as family 0 (log population, episode age, site and footprint food per meals, sustainable and absolute herd capacity, animals, last growth, direct daughters, fixed housing per resident). They are all current or past state, which is the required timing.
3. Add typed families, each a fixed row count with a presence mask and an overflow-aggregate row, all captured at the decision boundary before any action is applied:
   - Fields: per owned or operated tile, capital E, operating fraction, operator or access flag, current productive ecology, seasonal parameters, expansion availability, tool and maintenance demand.
   - Herds: per owned herd, exact count, species growth and death parameters, GrowthRemainder, DeathRemainder, food and water nutrition and deficit, management flag, shared forage and water capacity. Owned counts are already exact in the actor observation; the remainders and deficits are not.
   - Unlocks: era, acquired technologies, lifetime and Neolithic practice, exposure, effective progress against each threshold, withholding flags. The actor sees only capability flags.
   - Coupled economy: exact stocks by good, recipe and tool constraints, labor and population, food, clothing and housing need, and counterpart stocks for known trade partners (training-only, never published to the actor).
4. Emit the view on the same row as the observation it belongs to. A returned village row today pairs the reward of the executed transition with the next decision's observation; document that pairing in the row and keep privileged aligned with the observation, not the reward.
5. Add a headless check: with the privileged view perturbed and the observation held fixed, every actor-visible tensor and every legality mask is byte-identical.

### Learner (branch from feature/g0-pretrain 8baf309)

1. In bridge.py lines 226 to 244 stop dropping privileged. Build it in a separate critic-only batcher that returns its own dictionary; never merge it into the actor batch. Add an assertion in the actor forward that no privileged key is present.
2. In labor_model.py, extend LaborEncoderLayout with a versioned privileged block (widths per family, presence, schema version). In the three critic encoders project each privileged family through its own shared projector plus family embedding into the global attention set, so the 90 global tokens become 90 plus the privileged rows. Do not inject privileged state into the actor encoder class at all; make its constructor reject a nonzero privileged width.
3. Normalizer: the labor checkpoint already carries unused privileged mean, std and dead moments. Rebind them to the new schema version, freeze them at fit time, and audit the dead mask on real data before training (a named field can be silently zeroed).
4. Data: store the privileged arrays and schema version in teacher shards (labor_bc.py row builder near line 1012, labor_teacher_data.py writer), DAgger rows (labor_dagger.py lines 325 to 388) and search tensor rows (labor_search.py lines 1374 to 1427). In load_labor_datasets, a shard without privileged data is rejected for critic training, never zero-filled.
5. Checkpoint: a new architecture string or contract field, persisted model widths, a strict-load migration hook (precedents: the FiLM load hook and the value-support convert tool), and the privileged schema version inside labor_critic_digest.
6. Tests before any fit: actor-output invariance under privilege-only perturbation; critic output change under the same perturbation; schema-version mismatch rejected at load; normalizer dead-mask report.

### Data and acceptance

Re-collect fit data (60 worlds, 25 decisions) and calibration data (20 held-out worlds) once with the new emission; about 85 and 76 minutes at the observed rates. Fit the same three critics with the same support, epochs and sampling as critic3, plus the hygiene changes above, and run the existing critic-calibration evaluator. Report per head and per stratum against critic3 on the same worlds as the control. Acceptance is the existing G2C numbers on the QoL head that currently fails: normalized MAE at most 0.20, absolute bias at most 0.10, ranking at least 0.70, with the other three heads not regressing. If QoL still fails with privileged inputs, that result is the diagnosis the proposal lacks, and it points at targets rather than inputs.

## Intervention B: action-conditioned advantage head

Add a scalar advantage head per objective that scores a complete candidate action at a paused root, trained on within-root return differences, and use it only to rank search candidates before native re-evaluation. It is not an actor signal and it does not change the V heads or their gates.

### Architecture

1. Dueling form: Q_j(C, a) = V_j(C) + A_j(C, a) for the two task objectives (QoL-years, population-years) and the two costs. V_j stays the existing head (distributional for tasks, scalar for costs) and is not retrained by this change. A_j is a scalar linear output on a small MLP. A single scalar avoids the decode-noise problem: one 255-bin step is 0.043 symlog units while the median search gain is 0.012 normalized units.
2. Action encoder: build a fixed-layout semantic vector from the executed action contract: 14 specialty shares, 13 by 22 activity shares, movement slot one-hot (17), fission one-hot (3), seven trade gates one-hot (21), seven trade values (log scale, clipped) and the trade share. Bind rows to the current catalog order so the same specialty always lands in the same slot. Pass it through an MLP to a 128-d action embedding and concatenate with the critic's contextual village token and the privileged readout from intervention A. Token-level injection (action quantities as extra columns on specialty and activity tokens) is a later ablation, not the first cut, because it costs one full encoder pass per candidate.
3. Centering: at each root, subtract the mean A over the scored candidate batch before use, so A is an advantage relative to the candidate set and V stays the state baseline.

### Training data

The currently running search export cannot train this head: it stores only per-specialty counts and a payload digest, no activity, trade value or trade share, and no successor tensors. The next search run must store the full candidate payload per row, the per-step years, the re-evaluation returns and standard errors per stream, and the exact horizon reached. Keep the current export as a held-out ranking test set by reconstructing payloads only where the digest matches a stored candidate; otherwise discard those rows for training.

### Loss

Per root r with candidates i, k scored at the same horizon H and the same continuation stream:

L = sum over roots r, sum over pairs i<k of w_ik · softplus(−sign(G_i − G_k) · (A_i − A_k)), with w_ik = 1[|G_i − G_k| > 2 SE_ik]

G is the re-evaluated finite-horizon return normalized by the frozen reference scale, in physical-time discount. Pairs within two standard errors are excluded, never pooled across horizons, and never mixed with the single-rollout selection return when a re-evaluated return exists. Add a secondary regression of (A_i − A_k) onto (G_i − G_k) at weight 0.1 so magnitudes stay in return units. Split by world before building pairs.

### Validation, all preregistered

1. Held-out same-root pairwise ranking accuracy on pairs beyond two standard errors, per objective and for the personality-weighted sum.
2. False predicted improvement: the fraction of roots where the A-argmax candidate's re-evaluated return is below H12's by the margin.
3. Pruning regret: on a held-out subset where all candidates are natively evaluated, the gap between the best of the top four by A and the best of all.
4. Controls: a known-harmful candidate (all idle) must rank last; a delayed-benefit case (Farmer adoption on a suitable site) must not rank below H12 at horizon 3.
5. Paired-successor bookkeeping probe: two candidates differing only in bookkeeping fields must receive the same A within tolerance.

### Use

Inside labor_search, score all candidates with A, natively evaluate the top four plus H12 always, and keep the full-evaluation path on a declared fraction of roots to keep measuring regret. Record predicted A and realized return on every evaluated candidate so calibration keeps accruing.

### What it must not do

No actor loss term reads A. No PPO advantage is replaced by A. A never enters the actor batch, the actor gate or the G2C V criteria. A candidate's A score is never written as a critic outcome label. Candidates are sampled from the frozen actor and H12 as today, not selected by A before sampling.

## Numbers to preregister before any run

These are proposed defaults. Fix them in the manifest before the first collection, and change none after a result is seen.

| Item | Default |
|---|---|
| Privileged families and row caps | fields 16 rows, herds 8, unlocks 1, economy 1, each with presence mask and one overflow row |
| Value support | [-0.5, 2.5] by 255 bins, unchanged |
| Target tail | randomized tail as in the anchors (prefix 25, 16 draws, rho 0.95, cap 200) for every critic; otherwise an explicit age feature in all arms |
| Critic fit budget | same as critic3 (4 epochs, batch 256, lr 3e-4, grad norm 1.0), then a second arm at 12 epochs |
| Calibration worlds | 20 held-out, at least 12 and 2 per family; 500 groups |
| QoL head acceptance | nMAE at most 0.20, absolute bias at most 0.10, ranking at least 0.70; other heads no regression beyond 0.02 |
| Advantage head | 128-d action embedding, two-layer MLP, scalar output per objective |
| Pair exclusion | two standard errors of the re-evaluated return; same horizon and stream only |
| Ranking acceptance | at least 0.70 pairwise on held-out roots; false predicted improvement margin 0.01 normalized units at rate at most 0.15 |
| Pruning | top four by A plus H12; full evaluation kept on 20% of roots for regret |
| Seeds | local diagnostics at most 3 seeds; VM comparison five paired seeds; 3000 to 3059 sealed |
