# Labor actor personality conditioning (FiLM)

The specialty-labor learner (action 12) weights its two terminal objectives with a personality
`p = (q, p_pop)` on the simplex over `(qol_years, population_years)`. The combined actor advantage
is

    A = q * A_qol / s_qol + p_pop * A_pop / s_pop - sum_k m_k * A_cost_k

with the frozen reference scales `s` of `LABOR_OBJECTIVE_REFERENCE_SCALES`. Before this change the
personality was a single fixed checkpoint value (balanced 0.5/0.5) and entered the model only as
two extra columns of the village-token projection.

## FiLM in every labor encoder

`LaborObservationEncoder` (`labor_model.py`) is the encoder of the actor, of the shared two-task
critic and of each of the two cost critics (four separately parameterised encoders). Each now
carries FiLM conditioning ported from the contract-12 `TokenEncoder` design:

* condition `c = p - (0.5, 0.5)`;
* bias-free, zero-initialised `Linear(2, 2 * width)` maps give a per-channel scale `gamma` and shift
  `beta`, applied as `h * (1 + gamma) + beta`;
* applied before each local-activity attention layer (`film_local`, one per layer; the condition is
  repeated over the specialty-major activity groups), before each global attention layer
  (`film_global`) and before each of the two trunk linear layers (`film_trunk`);
* the village projection keeps its personality input unchanged.

Identity guarantees, covered by `tests/test_labor_personality.py`:

* at `c = 0` the output is bitwise the pre-FiLM computation, even with nonzero FiLM weights (the
  encoder keeps the exact pre-FiLM path as `encode(..., film=False)` for this check, and the test
  also compares against the committed pre-FiLM `labor_model.py` from git);
* FiLM modules are built under a forked CPU RNG, so for a given torch seed every historical
  parameter (including heads built after the encoder) initialises exactly as before;
* a load-state pre-hook fills zero FiLM weights for checkpoints saved before FiLM, so they load
  under `strict=True` and behave identically.

Optional `film_bound` (`bound * tanh(x / bound)` on `gamma` and `beta`) is off by default. It is a
conditioning field of the checkpoint contract, set by `initialize --film-bound`, and applied to all
four encoders on load (`set_labor_film_bound`). `labor_film_extent(module, tensors)` reports the
largest applied `|gamma|`/`|beta|` as an instability monitor.

Parameter cost per encoder (2 layers): `film_local` + `film_global` = 4 x 2 x 2 x token_width, and
`film_trunk` = 2 x 2 x token_width + 2 x 2 x width. At the production sizes this is +3,584 for the
actor and task-critic encoders (width 256, token width 128) and +3,072 for each cost encoder
(width 128): 13,312 parameters in total.

## Manifest-declared personality distribution

`labor_training initialize` declares the run's distribution, which is stored in the labor model
contract (`personalityConditioning`) and re-read by every later phase:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--personality` | `balanced` | `balanced` (fixed 0.5/0.5) or `dirichlet:ALPHA` |
| `--personality-key` | optimizer seed | keyed stream for every draw |
| `--personality-drift` | 200 | fission-daughter Dirichlet concentration (0 copies the parent) |
| `--film-bound` | off | optional tanh bound on FiLM `gamma`/`beta` |

The vmqueue manifest generator (`tools/vmqueue/labor_3x5_manifest.py`) takes the same
`--personality` and `--film-bound` and passes them to the `initialize` phase only. With the
defaults the emitted manifests and checkpoint contracts are byte-identical to before.

Sampling (`labor_personality.py`):

* each founding lineage draws `(q, p_pop) ~ Dirichlet(alpha, alpha)` once per world from the keyed
  stream `(key, iteration, world_seed, society, FOUNDER_TAG)`;
* a fission daughter (a new society whose `parentSociety` is known) draws
  `Dirichlet(200 * parent)`, keyed `(key, iteration, world_seed, daughter_society, DRIFT_TAG)`;
* `iteration` is the checkpoint iteration, so the rollout and the train anchor of one iteration
  share lineages while later iterations see fresh ones;
* keyed draws make the assignment independent of village order, rewinds, retries and phase order.

The per-world resolver is threaded through the BC collector (`personality_for_world` hook in
`labor_bc.py`, both the actor-BC and critic-pretraining collectors), PPO rollout and bootstrap
values (`collect_labor_rollout`), anchor/value collection and actor-gate evaluation
(`collect_labor_anchor_arrays`). Every row records its own personality in its
`labor_personality` tensor. `build_labor_training_batch` combines each row's advantage with that
row's recorded weights (a retry uses its parent's) through `combine_labor_advantages(row_weights=)`;
balanced rows reproduce the fixed balanced combination bitwise. A non-balanced distribution also
adds `personalityConditioning`/`personalityIteration` to the behavior signature, so rollouts,
anchors and gate reports stay bound to the exact assignment.

## Teacher labels carry no personality signal yet

The heuristic BC teacher is personality-agnostic. BC rows record their sampled personality, but the
labels are the same for every personality, so behavioral cloning teaches the actor to *ignore*
personality: expect the actor's FiLM weights to stay near zero after BC. Personality-conditioned
labels need search scored per personality (a later G3 item); they are not faked here.

Likewise the task critic predicts per-component returns (`qol_years`, `population_years`), which
are personality-independent under a fixed controller, and the cost critics predict
personality-independent costs. Their FiLM weights should stay near zero; that is expected, not a
defect. Personality becomes informative to the critics only once the actor's behavior depends on
it.
