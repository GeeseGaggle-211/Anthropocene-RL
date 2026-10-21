# model9: search teacher (expert iteration)

Status: built 20 October 2026, approved by the user. Simulator fork/restore (`ElderModel9Environment.Fork`/`Restore`,
6 headless assertions), bridge ops `fork`/`restore`/`drop`, label generation (`search_teacher.py`) and the imitation
loss (`ppo --search-labels`). First measurements are below.

## Why

PPO does not find gains that are known to exist. The `--model9-oracle` sweep found fixed mixed-use choices that beat
the heuristic in 5 of 6 worlds, by 1-11%. Learned movement never moved in the 2x2, because sampled moves carry low
advantage. Random perturbations of a strong heuristic are mostly worse, so the advantages for new behaviour are small
and noisy.

Search fixes this by trying alternatives in the simulator itself. It keeps the winners, and the policy imitates them.
The next search then starts from a better policy.

## Doctrine

- **Authoritative state.**
  - The simulator stays authoritative.
  - A fork saves the world plus each village's environment context: previous decision, route feasibility, fishing
    history.
  - Branches are discarded after scoring.
  - Labels are learner data, never simulator state.
  - `Fork` rebases the live environment on its own saved world, so the live path and every branch start
    byte-identical. Checks: a fork does not change the path; a restored branch replays it exactly; a deviating branch
    diverges; branches do not leak into each other.
- **Temporal abstraction.**
  - One decision per 20-year turn.
  - An alternative is a macro-action held for `horizon` turns (default 6, 120 years):
    - a region move, then keeping its intent;
    - a pure economy mix, held throughout;
    - the native heuristic, every turn.
  - Score: the village lineage's discounted welfare over the horizon. Credit is self-interested, as in the trade
    design: the lineage's own settlements, with daughters founded during the horizon included.
  - Whole-world welfare is recorded per branch, to catch lineage gains that are taken from neighbours.
  - The critic's end-of-horizon lineage value is recorded, and is added to the score only with `--bootstrap 1`; see
    the measurements below.
- **Paired comparisons.** Every other village follows the policy deterministically. Branches therefore differ only by
  the alternative, with no world or seed noise. This is the variance reduction the plain PPO gate lacks.
- **Scaling.**
  - Cost per label is (alternatives, at most 12) × horizon × turn cost, plus one save/load per branch.
  - Turn cost grows with active villages (about 14 ms per village-decision, `scale_probe.py`). There is no work
    proportional to simulated duration.
  - On Small worlds a branch costs a few seconds. At Standard/80 a turn costs 7-18 s, so one label would take minutes.
  - So search runs on Small worlds and valley fixtures, and the policy transfers. Large-world search would need
    fewer alternatives, a shorter horizon, or local sub-worlds.
- **RL implications.**
  - Labels come from policy k and train policy k+1 through an imitation loss inside PPO: cross-entropy on the region
    head, and soft cross-entropy on the economy mix. Only rows whose winner beats the policy by more than a margin
    (default 0.5%) are used.
  - Labels go stale as the policy changes, so each expert-iteration round regenerates them from the newest checkpoint.
  - The mix and region heads carry no teacher-KL anchor, so imitation does not fight the anchor.
- **Validation.**
  1. Fork determinism (headless).
  2. Label statistics: how often search beats the policy, by kind, and how much of each gain comes from the horizon
     return as opposed to the critic bootstrap. A gain only the critic sees is not trusted.
  3. Each round is gated against the frozen reference under the regression rules (architecture doc, *Output
     tokens*).

## First measurements (20 October 2026, laptop)

Policy `ppo10/stage2-span-seed0/best-single.pt` (stage 2, native movement). 16 worlds: TRAIN 1000-1009 and six
relaxed valley fixtures. 3 decision turns × 2 villages per world, so 96 decision points, each with up to 12
alternatives.

| Run | Horizon | Score | Branches (s each) | Search beats the policy (> 0.5%) | Winners |
| --- | --- | --- | --- | --- | --- |
| r0 | 6 | return + critic bootstrap | 874 (2.6) | 80% | region 75, native 1, mix 1 |
| r1 | 8 | return only | 867 (2.9) | 87.5% | region 61, native 22, mix 1 |

- **The critic bootstrap is not trustworthy here (r0).**
  - It is about 5.7× the horizon return, so it decides the score.
  - Native branches lost 15% on the bootstrap at about equal returns. The critic has only seen Elder-acting states,
    so heuristic-acting and moved states are off-distribution.
  - Return-only scoring is therefore the default; the bootstrap is recorded as a diagnostic.
- **Region moves are the main finding (r1).** Search beats the policy's own native movement at 61 of 96 points. The
  median lineage gain among winning labels is 13.6% over 160 years.
- **The gains are not taken from neighbours.**
  - The winners' whole-world welfare rises by a median of 0.8%. It is smaller than the lineage gain because the
    lineage is a fraction of the world.
  - Only 2.4% of winners lower world welfare.
- **Most moves are still worse.** The median region alternative equals the policy, and 1 in 10 lose more than 18% over
  6 turns. This is consistent with PPO's finding that random moves carry low advantage: the good moves are specific.
- **Mixes and the heuristic rarely win.** Pure-mix vertices are on average 0.4% worse than the policy, and the native
  heuristic 0.9% worse.
- **PPO path.** `ppo --stage mixed --learned-movement --search-labels r1/labels.npz` loads 84 label rows and applies the
  imitation loss (smoke iteration).

## Personality-scored labels (21 October 2026)

The user directed that each search label return the same value vector the critic uses, so every personality can
process it. `search_teacher --personality dirichlet:a` now works as follows:

- **Personalities.** Each society draws one personality. Founders draw from the distribution. Daughters copy their
  parent exactly inside branches, so branches stay paired; training adds drift.
- **Acting.** The policy acts with each village's personality in every branch (FiLM in the token model).
- **Returns.** Each alternative returns its lineage's discounted return per objective (`alt_returns`), in
  `contracts.OBJECTIVES` order:
  - welfare;
  - population-years;
  - realised trade utility;
  - the lineage's population-weighted QoL-years. This is one number per lineage, so splitting is not rewarded as more
    QoL.
- **Bootstraps.** The critic's per-objective, FiLM-conditioned end values are recorded as `alt_bootstraps`. They are
  added to the returns only with `--bootstrap 1`.
- **Score.** The searched village's label is scored for its society's personality `p`:
  `1 + sum_k p_k (R_k(alt) - R_k(policy)) / |R_k(policy)|`. At the default personality this is the welfare ratio to
  the policy, so earlier labels mean the same.
- **Imitation.** The label row stores `p`, so the imitation loss conditions the actor on the personality the label was
  scored for. Stored value vectors let labels be rescored for other personalities later.

## Round 3: probes, policy tails and critic branch shards (21 October 2026)

Built from the approved round-3 spec (`MODEL9_M7.md`, *Round 3 build*, lists the flags).

**Fission probes** (kind 4, `--fission 1`). The searched village submits `fission` Encourage (2) or Discourage (1)
on every turn of the horizon; daughters follow the policy. A probe equal to the policy's own learned code is not run
(it is the policy branch). The label stores `label_fission` (-1 on other rows), and the imitation loss adds
cross-entropy on the actor's `fission` logits for those rows.

**Trade probes** (kind 5, `--trade 1`; slot = probe id):
- 0 withhold (gate 0 every group), 1 offer stock (gate 1), 2 make to order (gate 2);
- 3 and 4: the relative value of the trade group with the largest holdings (`held`, the first of the bridge
  schema's `tradeFeatures`) times 0.5 and times 2, clamped to `TRADE_VALUE_RANGE`.

The base posture is the policy's own when it learns trade. Otherwise it is the heuristic's teacher label (tradeValue,
tradeGate, tradeShare) on the main-path step whose observation the fork restores; a restored step carries no teacher
labels, so the main path's is used. Without a label it is the native posture: values 1, every gate make-to-order,
share -1 (the configuration's share).

Every probe submits a **full posture** every horizon turn: `ElderTrade.From` fills an absent value or gate vector
with native values (1, make), and the bridge treats an absent share as native. A partial posture would silently mix
the probe with native trade.

Only groups present in the village's trade-set observation count as changed. A village with no trade row gets no
probe, and a probe that changes nothing is skipped. On the heuristic's native posture, that drops make-to-order, so
four trade probes run.

**Trade labels.**
- `label_trade_value` [7] and `label_trade_gate` [7] hold the winning probe's full posture.
- `label_trade_mask` [7] holds, per group, bit 1 if the probe changed its value and bit 2 if it changed its gate.
- The imitation loss touches only the masked components:
  - a Gaussian negative log-likelihood pulls `trade_value_mean` toward log(label). Its scale is detached, so imitation
    never narrows exploration;
  - cross-entropy on that group's gate.
- Probes inform the heads and never restrict them: unmasked groups, the export share and every scale get no
  gradient.

Earlier label files load beside new ones (missing round-3 arrays mean no fission or trade label; per-alternative
arrays are padded to the widest file).

**Policy tails** (`--tail 4`, 0 off). After each branch's horizon, the branch plays `tail` more turns of pure policy.
The tail never enters the score. Its turns become critic **branch shards** under `<out>/branch_shards/`, one directory
per branch (`<seed>_<withheld>_<warmup>_<point>_<alternative>`), in the teacher-shard layout `data.load` reads.

Each row is one lineage settlement's tail decision. It carries:
- the observation arrays, the privileged block and the personality it acted at;
- the outcome keys `data.reward_components` needs;
- label arrays with every mask false (the actor never trains on them);
- `world.npy`: a unique int64 world id per (seed, withheld, warm-up, point, alternative), above 2^56, so branch
  sequences never collide with each other or with teacher shards;
- `alt_kind.npy` (for validation);
- `founded_turn.npy`: the turn each settlement was founded on the full path. `data.lineage_rewards` uses it, so the
  daughter-sharing window is not restarted by a shard starting mid-episode;
- `bootstrap_index.npy` into the branch's `bootstrap/` block: the observation after the last tail turn of each living
  settlement.

Rows inside the held horizon are not policy rows and are not written. `branch_shards.load` concatenates directories
and rebases the bootstrap indices.

**Critic mixing.**
- `train --vector-critic` pretrains one head per objective. `--branch-shards` adds branch rows to that pretraining
  (critic only), with lambda-returns bootstrapped by the critic after each branch's last tail turn
  (`data.lambda_returns(..., bootstrap=)`; MC with the stationary continuation in the first round, as for teacher
  rows).
- `ppo --branch-shards` draws `--branch-share` (0.25) of each critic minibatch from branch rows. Their targets are
  recomputed every iteration with the current critic (`branch_shards.targets`).

**Validation** (`branch_bias.py`). On held-out branch shards, the tool compares the critic's prediction with the
realised tail return plus the discounted bootstrap (a lambda = 1 return). It reports relative bias per alternative kind
and head:
- per branch entry: the lineage's summed value at the first tail turn, the state the search scores;
- per row.

The spec's target is within 10% for native, region and mix.

The two numbers are not the same quantity. With a 4-turn tail, about 0.82 of the realised target is the critic's own
discounted bootstrap. This check therefore measures the critic's consistency over 80 years on search-reached states,
not its absolute calibration. The round-1 "5.7x" was the bootstrap's size relative to a 6-turn return, which is
expected to be large.

## Interfaces

- `python -m anthropocene_rl.autonomous_v7.search_teacher --checkpoint <pt> --seeds <train range> [--valley-seeds f]
  [--horizon 6 --points 4 --villages 2] --out <dir>` writes `labels.npz` and `summary.json`.
- `ppo ... --search-labels a.npz,b.npz --search-weight 0.5 [--search-margin 0.005]`. Region labels matter only with
  `--learned-movement`.
- The bridge's `verify_replay` refuses after a fork: verification does not carry the environment context.

## Limits

- Scores use welfare at the default personality. Personality-weighted search needs the vector critic's
  per-objective bootstrap. The horizon returns could be re-weighted from stored components; this is not built.
- Alternatives cover movement, the mix, native, fission policy and trade posture (round 3). Capital levels are not
  searched yet.
- Trade is rare on Small worlds (2 trades in 10 turns world-wide on seed 1000 under the heuristic), so most trade
  probes score exactly the policy's return. They cost up to 4 branches per point.
- The critic bootstrap may be miscalibrated off-distribution, for example right after a move. The summary therefore
  reports the return-only gain beside the scored gain.
