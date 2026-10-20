# model9: Elder personalities (preference-conditioned policy), design

Status: design, 20 October 2026, following the user's direction:
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
- `s_k` is a running scale of component k's advantage. Population-years are about 100 times larger than QoL-years,
  so unscaled weights would mean nothing.
- With the default `p`, the welfare-only advantage is reproduced exactly (`s_welfare` is not applied when `p` is one-hot
  on welfare).

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

## Simulator (later, with the game)

- A `Personality` field on `Society`, persisted and inherited at fission with small drift.
- An observation block so the native heuristic could also read it.
- Until then the learner supplies `p` alongside the observation; the simulator is unchanged.

## Order

Learner implementation follows the combined-curriculum stage. It touches the critic, the advantage and every run's
inputs, so it should not change a stage mid-flight. The first experiment is a personality sweep from one
Dirichlet-trained run against default-personality runs on the usual gates.
