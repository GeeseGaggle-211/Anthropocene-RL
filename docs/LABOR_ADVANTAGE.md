# Labor action-conditioned critic (intervention B) and bounded QoL head

Source: intervention brief 2026-10-26 ("Intervention B") and G0 manifest amendment r5-b (user-approved),
which overrides the brief where they differ. Code: `labor_advantage.py` (model, schema, fit, ranker),
`labor_advantage_eval.py` (preregistered validation), `labor_model.py` (`LaborQoLHead`, encoder `context`).

## Q = V + A

`A_j(C, a)` is a scalar per objective (QoL-years, population-years, feasibility, settlement guard), in
reference-scale units. Each objective has a 2-layer MLP over `[critic context ⊕ privileged readout ⊕
action embedding]`; the action encoder (349 → 256 → 128) is shared. The context is the frozen critic's own
encoder: the task critic's trunk output for the two task objectives and each cost critic's for its cost,
plus the presence-masked mean of the privileged family tokens after global attention (zeros for a critic
without a privileged block). The critic is never trained by this head: contexts are computed under
`no_grad`, and `fit_advantage_head` asserts the `labor_critic_digest` is unchanged. A is centred over the
scored candidate batch at each root before use (`centre`, `q_values`). The last layer is zero-initialised,
so an untrained head gives `Q = V`.

Default widths (task 256/128, cost 128/128): action encoder 123,008 parameters; each task head 131,585; each
cost head 98,817; total 583,812.

## Action vector (`labor-action-semantic-vector-v2`, 349 floats)

v2 (4 Nov 2026) keeps the v1 layout and changes two encodings: with the root `population`, the specialty
block is the share of residents after native largest-remainder apportionment, and the activity row of a role
with no apportioned residents is all idle. `action_layout_state()` records both (`specialtyEncoding`,
`unstaffedActivity`), so `load_advantage_head` refuses a head bound to the v1 encoding.

14 specialty shares, 13×22 activity shares, movement-union one-hot (17: stay, seven sites, keep, eight
regions; native −2 is rejected), fission one-hot (3), seven trade gates one-hot (21), seven trade values as
`log(clamp(v, 0.25, 4)) / log 4`, and the trade share. Rows bind to the catalog by stable ID: a payload may
declare `specialty_ids`, `activity_ids` and `trade_groups` in any order. Only the seven semantic fields are
read, so bookkeeping keys cannot move the vector.

`reparameterized_action_features` builds the same vector from one draw of the actor's output heads (ALR
Gaussian draws for labor, Gumbel noise for the categorical heads, Gaussian latents for trade values/share).
Its forward value is bitwise the encoding of the executed action: the draw is serialized by the actor's own
wire path (`schema12_initial_action`, labor simplexes canonicalized by `LaborPolicySample.payload`) and
encoded with `action_features(..., population=)`, so the specialty block apportions the serialized wire
shares, categoricals are the executed one-hots `argmax(logits + g)` (movement under its legality mask), and
trade value/share are the serializer's values. Every block is `hard + (soft − soft.detach())`: the gradient
is that of the continuous relaxation (ALR softmax, Gumbel-softmax at `temperature`, clamped log value,
batched sigmoid), never its value. The generator is consumed in `REPARAMETERIZED_DRAW_KEYS` order (unchanged);
`draws=` injects the noises instead and `return_draw=True` also returns them with the retained samples and
wire actions (`ReparameterizedDraw`). It is the Q-π path and is closed: unless it receives an open `QPiGate` (an evaluator report on exactly the reserved worlds
90154–90159 passing every gate, plus an explicit authorization), the actor outputs are detached and no
gradient reaches an actor parameter. Nothing in the repository opens it.

## Candidate-payload schema (`labor-advantage-candidates-v1`)

The consumer contract the search exporter writes to; the full field list is the module docstring of
`labor_advantage.py` and `parse_candidate_shard` is its strict reader. One `torch.save` dict per shard:
root observation tensors (no privileged key) and raw privileged arrays in separate dictionaries, and per
root its world/turn/camp/source, tensor row, personality, `full_evaluation` flag, optional `pair_sd`, and
candidates with the complete schema-12 action, `is_h12` (exactly one), optional `control` (`all-idle`,
`delayed-benefit`) and `bookkeeping_twin`, and evaluations `{horizon, reached_horizon, terminated, stream,
returns[4], step_years, se}` with raw discounted returns in component order. Stream 0 is the deterministic
selection stream; streams ≥ 1 are independent re-evaluations.

## Pairs and loss

Pairs are same root, same horizon, same stream kind, i < k. If both candidates have ≥ 2 shared
re-evaluation streams at that horizon, the paired mean difference and its paired SE are used; otherwise
the selection stream with its declared SE (per-evaluation `se`, else root `pair_sd`); one re-evaluated and
one single-rollout return are never mixed, and a pair with no declared noise is not comparable. Pairs are
never pooled across horizons. `w_ik = 1[|ΔG| > 2 SE]`. Loss per objective: mean over admitted pairs of
`softplus(−sign(ΔG)(A_i − A_k)) + 0.1 ((A_i − A_k) − ΔG)²`, summed over objectives. Horizon-specific
targets: by default QoL uses horizons 1–3 (predictable there, noise over 25 decisions per the QoL
diagnosis) and the other objectives every horizon (`--horizons`). Worlds are split (fit/held-out, 20%,
keyed) before pairs are built. The fit refuses reserved, calibration (90180–90199), development and sealed
worlds.

## Validation (`labor_advantage_eval`)

Held-out same-root ranking on pairs beyond 2 SE per objective and for the personality-weighted sums (root
personality and the 0.5/0.5, 0.9/0.1, 0.1/0.9 grid), with world-bootstrap intervals; false predicted
improvement (A-argmax realized below H12 by > 0.01 at the deepest common horizon); pruning regret on fully
evaluated roots (top 4 by A, and top 4 plus H12); all-idle ranks strictly last; delayed benefit not below
H12; bookkeeping invariance (data twins plus synthetic re-serialization, reversed ID declaration and extra
keys). Gates: ranking ≥ 0.70 (QoL, population-years, every utility; costs where they have pairs), false
improvement ≤ 0.15, both controls, invariance. Missing data is HOLD, not PASS. Evaluation is allowed on the
reserved worlds (`reserved_worlds_only` in the report) and refused on any world the head was fitted on.

## Search hook

`AdvantageRanker.plan(root_tensors, actions, h12_index=..., personalities=..., privileged_raw=...,
root_key=...)` returns centred A per candidate, the best-rank-over-personalities order, the indices to
evaluate natively (top 4 plus H12) and a keyed `full_evaluation` draw (default 20% of roots, then every
candidate). `RankerPlan.record` pairs predicted A with each realized return for calibration. Candidates are
sampled as today; A only chooses which samples are evaluated.

## Bounded QoL head (r5-b item 4)

`LaborQoLHead` (contract field `criticQoLHead`, version `labor-qol-bounded-discounted-mean-head-v1`)
replaces the symlog two-hot for QoL only: the head emits one pre-activation z, `m = sigmoid(z)` (default) or
`clamp(silu(z), 0, 1)` (fallback), and decodes `G = m W` with `W = dt / (1 − e^(−dt/τ)) = 410.1` QoL-years,
the discounted return of QoL = 1 forever (every closed target satisfies 0 ≤ G ≤ W because the per-step
reward is lineage QoL × years). Logistic fits a Bernoulli cross entropy against `clamp(G / W, 0, 1)`; the
fallback fits the squared error of the unclamped SiLU. Population-years keeps its distributional head.
Targets: the existing randomized-tail Monte Carlo return (default) or TD(λ) through the head
(`logistic,td-lambda,0.8`; pending user decision). Every fit reports the saturation monitor (share of
|z| > 6) at start and end. Select it with `labor_training initialize --qol-head logistic`. Checkpoints
without the field load and save unchanged (tested against the base commit's module).

**Head scale (user decision 27 Oct).** Closed returns lie in [0, W], but the randomized-roulette tail
is unbiased and unbounded: its weights are γ/ρ = 1.0013 > 1, so single tail estimates exceed W (up to
1.54 W in fit8-b04 world 90129), and a (0, W) head silently clamped them, biasing high-QoL worlds
down. The head now carries a range in units of W, `scale` (contract key `qolHeadScale`, serialized
only when it is not 1.0): `V = scale W sigmoid(z)` (SiLU fallback: `scale W clamp(silu(z), 0, 1)`),
fitted against `p* = clamp(G / (scale W), 0, 1)`. The clamp remains as a guard; the fit summary
(`qol_clipped_targets`) and every TD refresh (`diagnostics.clipped_targets`) count the targets it
would move, zero on current data at scale 2. `initialize --campaign-r5` with a bounded `--qol-head`
declares scale 2.0 (`LABOR_CAMPAIGN_R5_QOL_HEAD_SCALE`); `--qol-head-scale S` overrides it. An absent
field is scale 1.0, the historical head, bit for bit (checkpoint bytes, decode and loss), so every
old checkpoint loads and behaves exactly as before; a scale mismatch is a different contract and is
refused at load. Calibration metrics (nMAE, bias), the TD residual reference scales and the gates
stay in their units (W and the QoL reference scale); only the head's internal range changes.

**Separate QoL encoder (user decision 28 Oct).** On the shared task encoder the QoL head starved:
population-years took about 98% of the raw encoder gradient even at a x30-100 QoL loss weight, and QoL
underfit in-sample (nMAE about 0.5). Contract key `qolEncoder: "separate"` gives `LaborTaskCritic` a
second encoder, `qol_encoder`: a full copy of the task encoder architecture (privileged token path
included) with its own parameters. The QoL head reads only it; population-years keeps `encoder`. The key
is serialized only for "separate"; an absent key is the historical shared critic, bit for bit (state
keys, initial values, forward, a monitored weighted privileged fit and the contract bytes were compared
against a17abc5). `initialize --campaign-r5` declared it until 28 Oct; it now declares the shared
encoder again (Fable audit: no nMAE gain, lower held-out within-trajectory ordering, a negative level
offset, +34% parameters). The r5 baseline is shared + `--qol-loss-weight 30`; `--qol-encoder separate`
selects this encoder explicitly.
Initialization: an independent fresh encoder, built last under a forked CPU RNG seeded by
`labor_qol_encoder_seed(optimizer_seed)`, so no historical parameter or the global RNG stream moves and
`labor_critic_support convert-qol-encoder --init fresh` reproduces the same tensors from a shared
initialize checkpoint; `--init copy` instead copies a trained task encoder (warm start, outputs
unchanged). Fitting: the optimizer, resume state and stability blocks include `qol_encoder`; there is
no shared-trunk dominance detector (each encoder serves one head; the task critic stays a global-clip
contender, and both encoder norms are logged as `grad_norm_preclip`); `--qol-loss-weight auto` is
refused (no shared encoder share to balance); `--dominance-excess-loss` is accepted but has no
detector to act on (warning; summary `qol_encoder.dominance_excess_loss_effective: false`);
privileged attention and dead features probe `task` and `task_qol`. The advantage head reads, per
task objective, the context of the encoder feeding that objective's value head (QoL: `qol_encoder`;
population-years: `encoder`); widths are unchanged and the A artifact records
`task_context: per-objective-value-head-encoder`, checked at load.

**GradNorm loss balancing and per-encoder clipping (user decision 29 Oct).** At 60 worlds every fixed
QoL weight tripped `gradient_dominance` at seed 0 (weight 30: `task_encoder`, population-years dominant,
QoL progress 0.0012, step 1875; weight 50: `global_clip`, the task critic dominant, step 2225).
`pretrain-critic --loss-balancing gradnorm` (Chen et al. 2018) learns the two task-head weights on the
weight of the shared task encoder's last trunk `Linear`: every optimizer step measures the unweighted
head gradient norms g_i on that layer, G_i = w_i g_i, loss ratios L_i(t)/L_i(0) (L_i(0) = the first
step's batch losses; the QoL excess BCE under `--dominance-excess-loss`, else the raw head losses) and
relative inverse training rates r_i; the target is mean(G) r_i^alpha (`--gradnorm-alpha`, 1.5), held
constant, and the weights' own Adam (`--gradnorm-lr`, 0.025) steps on sum |G_i - target_i|, then floors
them at 1e-3 and renormalises them to sum to 2. `--qol-loss-weight`/`--task-loss-weights` give the
initial ratio (30 starts at 60/31 : 2/31). The weights a step uses come from the previous update; a
rolled-back step leaves them unchanged. The weights, L_i(0) and the weight optimizer are in the resume
payload (`gradnorm`) and the fingerprint; each stability step logs `task_loss_weights` and a `gradnorm`
record (norms, rates, targets, `weights_next`); the summary lists them under `task_loss_weights`.
Refused with `--qol-loss-weight auto` and with a separate QoL encoder. Under gradnorm,
`gradient_dominance` is advisory (non-mandatory, alarms in `advisories`) and the mandatory, early-stopping
`task_progress_stall` gates: any task head whose relative progress over the 16-point dominance window
(the dominance check's `stalled_progress` computation) is below `dominance_min_progress` (0.01)
(PREREGISTERED `stability_task_progress_stall`; the seed gate reads it like every mandatory check).
`--grad-clip per-encoder` clips the task critic (task encoder and heads) and each cost critic (its own
encoder and head) to `--max-grad-norm` separately instead of one global clip; step records add
`grad_norm_group_preclip` and `clip_multiplier_groups` (`grad_norm_global` stays the global pre-clip
norm). Both default off; the default fit is unchanged bit for bit (parameters compared against 50caf4c).
Through the acceptance harness both are plain `--critic-args` (shell `CRITIC_ARGS`).

Saturation keeps |z| > 6 (frozen r5 text). The expected-low stratum is a physical statement (G < 0.05 W:
no housing or clothing yet), so it is still applied to G/W, which at scale 2 is `p* < 0.025` in head
units; a low-saturated row decodes below `2 sigmoid(-6) W = 0.0049 W` (0.0025 W at scale 1), still far
inside that stratum. Reports of a widened head record `head_scale` and `expected_low_head_fraction`.

**Tail-factor diagnostic (advisory).** One frozen stop schedule per world is shared by all its camps,
so a world's tails move together. `critic-calibration` reports (and the r5 acceptance receipt, as
`advisory:qol-tail-factor` next to the QoL gates) give per world the schedule factor
`20 mean_d sum_{k <= stop_d} (γ/ρ)^k / W` (from `stop_indices` in the meta's `return_tail_record`; 1 in
expectation, 0.61-1.67 on the pilot data), the share of rows whose tail exceeds W, and the tail's weight
in the Monte Carlo targets `c_i tail / G_i` (`c_i` the discount from the row to the cutoff). The QoL
bias is given on all rows and on rows whose tail weight is at most 0.5. Never gating; its status is
REPORT.
