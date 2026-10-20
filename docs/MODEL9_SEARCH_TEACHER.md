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

## Interfaces

- `python -m anthropocene_rl.autonomous_v7.search_teacher --checkpoint <pt> --seeds <train range> [--valley-seeds f]
  [--horizon 6 --points 4 --villages 2] --out <dir>` writes `labels.npz` and `summary.json`.
- `ppo ... --search-labels a.npz,b.npz --search-weight 0.5 [--search-margin 0.005]`. Region labels matter only with
  `--learned-movement`.
- The bridge's `verify_replay` refuses after a fork: verification does not carry the environment context.

## Limits

- Scores use welfare at the default personality. Personality-weighted search needs the vector critic's
  per-objective bootstrap. The horizon returns could be re-weighted from stored components; this is not built.
- Alternatives cover movement, the mix and native. Fission, trade posture and capital levels are not searched yet.
- The critic bootstrap may be miscalibrated off-distribution, for example right after a move. The summary therefore
  reports the return-only gain beside the scored gain.
