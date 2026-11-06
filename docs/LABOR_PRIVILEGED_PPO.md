# Privileged PPO plumbing and bounded critic warm-up

This integration is based on `90eee94bcc124ba49482cd75fb57235fc3afa7cc`.
It connects the existing privileged **state-value** critics to labor PPO and
records sampled action inputs for subsequent state-action integration. It does
not add a Q model, train the ranking advantage head, open the Q-pi gate, or
change the target estimator or economic behavior.

## Existing training semantics

The r5 acceptance defaults are 25 trajectory decisions and
`logistic,td-lambda,0.8` for the QoL value head. A decision spans 20 years:
the prefix covers 500 simulated years. The discount time constant is separately
400 years (`exp(-elapsed_years / 400)`). `qol_td_lambda_targets` computes
`G_t = r_t + gamma_t ((1-lambda) V_{t+1} + lambda G_{t+1})`, using the
declared randomized tail or terminal boundary. These definitions are unchanged.
The G0 manifest's later section 11 increases the recorded teacher window to
50 decisions plus its tail; the CLI default is not evidence that a particular
trained artifact used only 25 decisions.

At this revision, `LaborTaskCritic.forward(tensors, privileged)` and
`LaborCostCritics.forward(tensors, privileged)` have no action input. The
separate `LaborAdvantageHead` consumes frozen critic contexts plus semantic
action features and is trained for candidate ranking. Its candidate-centering
contract is not silently substituted for the PPO state baseline. The exact
intended state-action model/checkpoint identity still needs to be established
before that part of the requested integration can be completed. The requested
contract is a state-action value fitted to GAE-derived return targets, with
ranking used as an evaluation metric. That is not established by the current
pairwise-ranking fit implementation. The local `g0/sa-advantage-check.md`
identifies its evaluated `scaleout-gnfull/critic` lineage as state-only; this
does not establish that a state-action artifact is absent elsewhere. A
short-horizon ranking diagnostic does not describe the long value-training trajectory.

## Data and gradient boundaries

Rollout rows retain raw privileged arrays separately from actor observations,
with a pinned privileged-normalizer digest. Initial inference, retry inference,
successor bootstrap and TD loss use the bound frozen normalizer. Pure validation
does not advance simulated time: retry privilege is copied from the same
pre-action row, while its observation includes the typed rejection context.
Successor privilege is read from the successor village only for the tail value.
Partial or mismatched privilege, schema, masks and normalization are rejected.

The persisted batch also retains detached full actor outputs and versioned,
349-dimensional semantic vectors of the exact submitted schema-12 requests.
Retry vectors combine the resampled labor request with the retained initial
movement/fission/trade request. These are policy action requests; native
fallback or physical resource consumption is part of the environment response,
not relabeled as the sampled action. Latents, masks and behavior log-probabilities
remain the likelihood source. The new action vectors are diagnostic inputs;
they are not currently used in value fitting or PPO advantages.

`labor_action_adapter` validates the submitted request against the catalog and
uses the existing fixed semantic encoder. Identical schema-12 requests from
actor, heuristic or search producers give identical features. The production
heuristic converter maps resident counts to population shares and worker-years
to fractions of assigned resident time; the actor serializer decodes masked
labor samples, exponentiates trade-value latents and applies sigmoid to trade
share. For example, actor latent `log(3)` and heuristic trade request `3` agree.
Wire trade requests `1.5` and `3` are different actions and are not multiplied
to match. Different requested shares that happen to compile to the same physical
result can also remain different inputs under this existing request-based
contract. No batch-statistic fitting or normalization change is introduced.

New collection artifacts include the full adapter contract and its digest;
replay rejects mismatches. The existing 349-feature layout remains unchanged,
and older rollouts without the additional adapter metadata remain readable.

Independent-return anchors now store separate raw privilege for parent and
ranking-successor rows, bound to the same normalizer digest. Anchor training
and calibration normalize those rows with that binding. Older unprivileged
anchors/checkpoints remain supported. Older anchors lacking privilege cannot
be used with privileged critics. Anchor targets remain independently observed
returns, never a warm-up bootstrap masquerading as calibration evidence.

## Bounded warm-up

`labor_training collect --warmup-decisions N` accepts 1 through 9 and overrides
`--turns`. The budget counts physical world decisions **per world**, not camps,
zero-time retries, or optimizer steps. For example nine decisions with two
living camps yield eighteen parent rows. Rewinds are refused in this mode.
Collection stops at the decision budget or world extinction with the actor fixed.

At the living boundary, all four targets bootstrap from the frozen source
checkpoint's privileged state value at the actual successor. Terminal rows
have no bootstrap. The existing two-attempt GAE keeps unit discount and trace
continuation on the internal retry edge, and charges physical reward once.
The boundary is an explicitly biased value bootstrap, not a complete 500-year
observed return or a claim that nine decisions establish convergence.

The saved rollout is marked critic-only. Passing it to `train` requires the
existing independent `--anchor`, rejects `--calibration-report`, performs only
the explicitly configured finite `--critic-epochs`, and checks that actor
parameters did not change. Actor optimizer state is preserved. The checkpoint
records the decision budget/count, tail policy and `readiness: false`.
It does not auto-stop on a readiness metric or open the actor gate. After
warm-up, collect a fresh rollout and obtain a fresh four-head calibration
report against the updated critic digest before an actor update. A prior gate
or rollout must not be relabeled with a new digest.

Example configuration only (no training runs were launched for this change):

```text
python -m anthropocene_rl.autonomous_v7.labor_training collect \
  --checkpoint initial.pt --worlds <approved-fitting-worlds> \
  --warmup-decisions 9 --out warmup-rollout.pt
python -m anthropocene_rl.autonomous_v7.labor_training train \
  --checkpoint initial.pt --rollout warmup-rollout.pt \
  --anchor independent-train-anchor.npz --critic-epochs <explicit-count> \
  --out warmed.pt
```

## Independent critic-seed candidates

Three eligible pretrained critic seeds can later seed three independent PPO
candidates, each loading the same initial actor and its own task/cost critic,
optimizer state, RNG state, output directory and checkpoints. No selection or
training is authorized by this plumbing. A shared frozen trajectory dataset
is possible only with identical actor/normalizer/action contracts; each
candidate must recompute its own values, bootstrap, advantages and critic-digest
binding. The current CLI deliberately rejects another checkpoint's rollout.
It does not yet orchestrate cross-checkpoint sharing or choose candidates.

## Validation scope

CPU fixtures cover privilege alignment through initial/retry/bootstrap and
serialization, masks and stale normalizers, finite action vectors, nine-decision
counting, anchor loading/loss/calibration, actor isolation during critic fitting,
fresh collection followed by a synthetically gated actor update, and independent
checkpoint storage. Model checkpoint tests use synthetic current/legacy schemas.
The cached production `scaleout-gnfull/critic/privileged-e4/seed-0/critic.pt`
also strict-loads with its fitted privileged normalizer (SHA-256
`fbe69fe4ec0d2e6466c9497d56fea3bf1fa946e41f36ea076ff5238f1bc6e5c8`).
Its metadata declares 50 decisions, randomized teacher continuation, a 400-year
discount time constant and QoL TD-lambda 0.8. It contains state-value critic
weights, with no action-conditioned head. The intended pretrained state-action
artifact has not been identified; no fresh action head is silently initialized
in its place. Fitted state-action values driving detached PPO advantages remain
the requested behavior, not a behavior completed by this adapter-only change.
The actor and critics here are feed-forward: there is no recurrent hidden state.

The real-actor update fixture exposed an existing masked categorical entropy
gradient bug (`0 * -inf` computed before `where`). Masking the log probability
before multiplication preserves finite entropy and gradients, including when
the entropy coefficient is zero. This is required for a subsequent PPO step.
The cached native bridge passes the one-world, one-turn normalizer/initialization
smoke (`test_fit_normalizer_cli_and_initialize_e2e`) when selected through
`ANTHROPOCENE_MODEL9_BRIDGE`. Full native PPO/state-action training was not run.
Local tests are diagnostic only and establish no training readiness.
