# model9: Elder personalities (preference-conditioned policy), design

Status: implemented in the token model, 20 October 2026. The pieces:
- `contracts.PERSONALITY_WIDTH` and the `qol_years` component;
- the village token reads `t["personality"]`;
- `ppo --personality default|fixed:...|dirichlet:a`, drawn per (world, lineage) and inherited by daughters;
- `combine_personal` for the per-row, scale-normalised advantage;
- evaluation at a given personality, and `personality_sweep.py`.

Personalities need the token model (the per-set-pooling network rejects them). An end-to-end smoke run (token clone,
Dirichlet PPO, sweep) passes. The design follows, written for the user's direction:
- Elders should have different personalities; some value QoL over population and vice versa.
- The mechanism should scale.
- It is "less a reward function change than a personality vector".

## Idea

One policy serves every personality.
- A **personality vector** `p` is a non-negative weighting over the objective components, normalised to sum to 1.
- `p` is an *input* to the actor and the critics. The village's reward is the p-weighted sum of its components.
- During training each lineage draws its own `p`, so the network learns the whole family of preferences. This is
  preference-conditioned (multi-objective) RL.
- At play time `p` is a trait of the Society: set at founding, inherited by daughters with drift.

This scales:
- A new objective (production, military strength, resource output) is one more vector-critic head and one more
  coordinate of `p`, with no new policy.
- Behaviour between personalities interpolates smoothly, so designers can place elders anywhere on the frontier.

## Components (initial)

| Component | Per decision | Notes |
| --- | --- | --- |
| `welfare` | P x Q_lineage x dt | today's reward |
| `population_years` | P x dt (living) | headcount |
| `qol_years` | Q_lineage x dt (living) | quality regardless of headcount |
| `trade_utility` | own realised trade utility | contract 11 |

- `qol_years` is new. It isolates the quality term that welfare multiplies by population.
- The default personality is welfare only, which reproduces every current run exactly.
- E2's QoL-for-population drift is the welfare personality's own trade-off. A QoL-leaning elder would make the
  opposite one.

## Learner

**Inputs.** `p` (4 values) is appended to the core input of the actor and of every critic.
- The upgrade adds zero columns, so existing checkpoints are exact under the default `p`.
- The vector critic's heads estimate each component's value *given* `p`, because behaviour depends on `p`.

**Advantage.** The per-row advantage is `sum_k p_k A_k / s_k`.
- Population-years are about 100 times larger than QoL-years, so unscaled weights would mean nothing.
- Since round 3, `s_k` is a fixed reference scale (`--personality-scales reference`, the default); see *Stable
  scales* below. Earlier runs used each component's batch standard deviation (`--personality-scales batch`).
- With the default `p`, the welfare-only advantage is reproduced exactly under either rule.

**Training distribution** (`--personality`):
- `default` (welfare one-hot);
- `fixed:w1,w2,...`;
- `dirichlet:a` (per lineage, around the default with concentration `a`).

A daughter keeps its parent's `p` within an episode.

**Evaluation.**
- Gates run at the default `p`, so they stay comparable with every earlier result.
- A **personality sweep** evaluates a grid of `p` and reports each one's population, QoL, welfare and trade.
- It checks that the frontier is monotone: more QoL weight never lowers QoL.

**Guard.**
- The no-collapse guard applies at every evaluated `p`.
- A QoL-leaning elder must not extinguish its lineage.
- A population-leaning elder must not breach the below-30 or QoL-drop floors beyond its declared trade-off. Those
  floors are per personality; the hard floors stay global: extinctions and regime viability.

## FiLM conditioning (21 October 2026)

The personality originally entered only as 4 of about 450 inputs to the village token's projection. Behaviour
cloning only ever showed the default, so the network starts out ignoring it.

The token encoder now also applies FiLM:
- `c` is `p` minus the default personality.
- Each attention layer's input tokens become `x · (1 + W_s c) + W_b c`. The village-token output after each trunk
  layer gets the same.
- The FiLM weights are bias-free and start at zero.

This gives three properties:
- **Exact for old checkpoints.** A load hook fills missing FiLM weights with zeros, so every checkpoint without them
  behaves as before.
- **Inert at the default personality.** With `c` = 0, outputs are unchanged and the FiLM weights get no gradient.
- **Scalable.** A new trait vector widens `c` rather than adding a token that every other token must attend to. The
  user preferred this for that reason.

The critic and cost critic use the same encoder, so their value estimates are FiLM-conditioned too.

## Welfare leakage (20 October 2026)

A personality run's policy follows the personality-weighted advantage, not welfare. Welfare still reaches it through
four side routes, and each one pulls personalities toward the default. Two of them are now closed.

| Route | Fix | Status |
| --- | --- | --- |
| Checkpoint selection | `spread_selection`, below | built |
| Shaping costs | `shape_components`, below | built |
| Teacher-KL anchor and the stage-1 clone | a weaker-anchor arm in the next stage; report each personality against the heuristic | planned |
| Global guard floors | floors per personality; extinctions and regime viability stay global | designed |

**Spread selection.**
- At each periodic evaluation, a personality run evaluates the checkpoint at three points: welfare `[1,0,0,0]`,
  population `[0,1,0,0]` and QoL `[0,0,0,1]`. Each point uses the same selection worlds; this triples the evaluation
  cost.
- Score: `min(log(QoL at qol / QoL at population), log(population at population / population at qol))`. Both
  directions must separate, and 0 means none.
- Eligibility: every point keeps extinctions at or below the init's, and the welfare point keeps every regime's
  welfare ratio at or above 0.97.
- The trajectory logs the score, both gaps and each point's outcomes under `eval.spread`.
- Default-personality runs keep the geometric-mean welfare rule.

**Per-component shaping costs.**
- The mix-change and break costs are a share of the decision's reward that turn.
- In personality runs the same share is charged on every component in its own units. A negative trade utility is
  not enlarged.
- Default-personality runs keep the costs on welfare only, so earlier runs reproduce exactly.

## Stable scales (round 3, 21 October 2026)

**Why.** The batch rule divided each objective's advantage by its batch standard deviation. The exchange rate
between objectives then drifted with batch content: the stage, the regime mix and the other personalities in the
batch. A `[0.5, 0.5, 0, 0]` Elder meant something different in a capability batch than in a mixed batch.

**Rule.** `A = sum_k p_k A_k / s_k_ref`, with `s_k_ref` fixed (`contracts.OBJECTIVE_REFERENCE_SCALES`).
- `ppo.update` standardises the combined advantage afterwards, so only the ratios between the `s_k_ref` matter.
- A welfare-only personality therefore reproduces the batch rule's update exactly (tested).
- **Clip** (`--personality-clip`, default 5, 0 off): each normalised component `A_k / s_k_ref` is clipped to
  `+/- c` before weighting. Reference scales measure return spread, but advantages also carry critic error. In the
  first smoke an untrained trade head's advantages were about 1,500 reference spreads wide (`scale_drift`), and with
  no clip any trade weight made that noise the whole advantage. The clip bounds a component's term by `p_k c`.
  Each iteration logs the clipped share per objective (`scale_clipped`) next to `scale_drift`. With the clip on, a
  welfare-only personality matches the batch rule except on rows beyond 5 welfare spreads.
- `--personality-scales batch` restores the earlier rule (no clip). The choice is saved in checkpoints (`personality_scales`).
- Each iteration logs `scale_drift`: each objective's batch advantage spread over its reference spread. Unequal
  drift across objectives is how far the batch rule's exchange rates would have moved that iteration.

**Measurement** (`scale_reference.py`):
- The native heuristic (no actions submitted) on TRAIN seeds 1000-1059, 45-turn rollouts, Small world, through the
  bridge.
- One row per village decision, with the fields `data.reward_components` reads.
- Per (world, camp) sequence, the discounted return per objective with the `HORIZON_YEARS` discount `ppo.gae` uses.
- The heuristic has no critic, so returns stop at the rollout end. Only rows with at least 20 later turns (the first
  25 turns) are kept, so every return covers a full 400-year-equivalent window.
- `s_k_ref` is the standard deviation of that return over the kept rows.

**Measured values** (21 October 2026; simulator f06e3be, contract 12):

| Objective | `s_k_ref` now (45 turns, full-window rows) | Mean return | First build (25 turns, all rows) |
| --- | ---: | ---: | ---: |
| welfare | 59,319 | 136,573 | 37,864 |
| population_years | 85,134 | 172,285 | 50,391 |
| trade_utility | 26.68 | 6.35 | 17.55 |
| qol_years | 43.15 | 259.0 | 64.0 |

- The kept rows are 37,085 of 82,841 (1,390 s on 12 bridges).
- The first build's all-rows values mixed in truncated late returns. Truncation shrank the three objectives that grow
  with the village (welfare, population, trade) and widened `qol_years` (short late returns against long early ones).
- In the new values, one QoL-year of `qol_years` weight is worth about 1,375 person-QoL-years of welfare weight; a
  unit of `trade_utility` about 2,200.

## Daughter-start bank (round 3, 21 October 2026)

Young daughters are where personalities and fission policy matter most, but an ordinary rollout reaches them late
and rarely. The bank stores worlds at the moment a daughter is founded, so later runs can start from them.

**Capture** (`ppo --daughter-bank <dir>`):
- When a village row first appears with `founded` true, it is offered to the world's reservoir sample of 3 over
  all of that rollout's founded daughters (algorithm R, keyed by the world, so deterministic). A kept daughter's
  world is captured then (bridge op `capture`; the daughter's first decision state), and the entry it displaces is
  deleted. Captures therefore spread over the whole episode. The first build kept the first 3, which put most
  entries at turns 10-12.
- Each capture is a zlib-compressed snapshot file plus an `index.jsonl` line: id, seed, withheld mask, movement,
  turn, the daughter's society and camp, and its parent society.
- The bank keeps at most 500 entries, oldest first out. It persists across iterations and runs.
- Entries with `id % 10 == 0` are a held-out slice for `daughter_eval`; training never draws them.

**Training.**
- With a non-empty bank, `world_sampler` makes slot `i % 4 == 0` a bank world; the valley slot stays `i % 4 == 2`.
  Stage 2 draws only capability-fixture entries. An empty bank changes nothing, the random stream included.
- `collect` resets a bank world from its snapshot (bridge `reset` with `snapshot`), under the entry's own mask.
  Regime bookkeeping uses the entry's seed and mask.
- Personalities: founders in a bank world draw fresh personalities, and daughters (the banked one included) inherit
  from their parent with drift, whatever the village order.
- A snapshot holds the world, not the bridge's per-village context. After a restart, no active commitments carry
  over and no village is reported `founded` on the first step.

**Search** (`search_teacher --daughter-search`, default 1): at a decision point with a young daughter
(`parentSociety` > 0 and first seen within the last 3 turns, or the banked daughter of a bank start), one of the
searched villages is a young daughter. Without one, the draw is the earlier uniform one. `--daughter-bank` with
`--bank-worlds N` also searches the newest N training entries from their snapshots.

**Evaluation** (`daughter_eval.py`): from each held-out snapshot, 5 turns with the checkpoint and with the
heuristic. Reports the daughter society's survival, population growth (final residents over the daughter camp's
residents at capture) and population-weighted QoL, paired per entry.

## Simulator (later, with the game)

- A `Personality` field on `Society`, persisted and inherited at fission with small drift.
- An observation block so the native heuristic could also read it.
- Until then the learner supplies `p` alongside the observation; the simulator is unchanged.

## Order

Learner implementation follows the combined-curriculum stage. It touches the critic, the advantage and every run's
inputs, so it should not change a stage mid-flight. The first experiment is a personality sweep from one
Dirichlet-trained run against default-personality runs on the usual gates.
