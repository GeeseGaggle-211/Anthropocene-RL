# G3 labor search teacher (action 12)

Status: built 25 October 2026 (`labor_search.py`, `tests/test_labor_search.py`), local smoke only.
Spec: brief §4.5, §8.4 and gate G3. Port of the contract-12 search teacher (`search_teacher.py`,
`MODEL9_SEARCH_TEACHER.md`) to the specialty-labor actor.

## What it does

For a protected root (a decision state on a fitting world, 90100-90159 only) and one focal camp:

1. **Candidates** (at most `--candidates` M, default 16), each a complete schema-12 action, natively
   validated (`validate_labor`, pure) and checked against the pre-action movement mask. Invalid ones are
   dropped and recorded in the candidate table.
   - H12: the converted heuristic label (`convert_heuristic_labor_label` on the shadow heuristic step, as
     `collect_heuristic_world` does). It is always candidate 0 and the protected reference.
   - Actor (with `--actor-checkpoint`): the deterministic action and `--actor-samples` K samples (default 4)
     at the focal camp's root personality, keyed generator.
   - Structured single-block H12 perturbations, interleaved round-robin across families so a budget covers
     each family:
     - transfer: 1, 2 and 10% of residents for Gatherer<->Farmer, Gatherer<->Craftsperson,
       Gatherer->Builder, Gatherer->Weaver and the two largest roles. A newly staffed role gets its key
       activity.
     - activity: within a staffed role, 0.25 of role time from its main activity to each other legal activity,
       or the role's idle share to its main activity.
     - movement: stay, versus up to `--movement-candidates` legal region moves (keyed choice; there is no
       destination score to rank by).
     - fission: each other policy.
     - trade: gate on/off (0 versus make-to-order) per group with an executable opportunity, meaning a present
       trade row with stock held or recent imports, exports or realised utility.
   - Joint, fixed before any evaluation: movement + Gatherer->Farmer, Gatherer->Craftsperson + trade gate,
     fission + Gatherer->second role. Bests from different branches are never combined.

   Every labor perturbation conserves residents (the native largest-remainder apportionment reproduces the
   declared integer staffing exactly) and worker-time (each role's activity simplex sums to one).
2. **Successive halving over horizon.** Round r extends the survivors to `horizons[r]`. A step restores the
   branch's own fork, so each round adds only the new decisions. Step 0: the focal camp executes the
   candidate and every other camp its root H12 action. Later steps: every camp follows H12 from a fresh shadow
   heuristic step. Survivors are kept by their **best rank over personalities** (competition rank, keyed
   tie-break). H12 is always carried as the reference. Default: all M at h=1, 1/2 at h=2, 1/4 at h=3, 2 at h=5.
   `--max-resolves` caps the search resolves per focal root; a round that would exceed it is skipped and the
   cut is recorded.
3. **Scoring.** The focal camp's discounted components (tau = 400 years), credited as the labor reward code
   does: `_transition_rewards` (lineage-shared QoL-years, population-years), the initial-attempt feasibility
   cost, `_settlement_guard`, and the `discounted_returns` recursion (zero after termination). The search
   scalar is `U_p = w_q G_q/s_q + w_p G_p/s_p - eta_f C_f - eta_s C_s`, with the frozen `labor_model`
   reference scales and the PPO default multipliers (1, 1). The lineage aggregate used by the contract-12
   teacher (population-years over the focal lineage, daughters included, and population-weighted QoL-years)
   is recorded as a diagnostic only. Scoring with it would credit fission differently from PPO.
4. **Personalities from one set of rollouts.** Return vectors are stored once and scored under the grid,
   by default balanced, (0.9, 0.1) and (0.1, 0.9), plus the focal society's root personality when
   `--personality dirichlet:a` is declared.
5. **Selection and independent re-evaluation.** Per personality, take the best found among the candidates
   that reached the final horizon; ties go to H12. Only winners that beat H12 by more than the margin on the
   selection stream are re-evaluated. The winner and H12 are rerun from the root on `--reeval-streams`
   (default 3) fresh streams. Accept when the paired mean gain exceeds `margin*|U(H12)| + 2 SE`
   (default margin 0.5%). Otherwise the label stays H12.

## Continuation streams (deviation to review)

The native simulator and the converted H12 controller are deterministic. Restoring a fork and replaying
reproduces the physical successor exactly; `verify_exact_restore` checks this on every world's first root
and records it. Every selection branch at a root therefore shares one exact continuation (stream 0), and
"fresh seeds" would only repeat it.

Independent re-evaluation uses a declared stochastic controller, `h12-dither`. It is H12 where each camp,
with probability 0.5, moves 5% of residents between Gatherer and one staffed role. The dither is natively
validated and reverted when rejected. Draws are keyed by (world, root turn, focal, stream, step, camp), so
the winner and H12 get common random numbers. The focal camp's own first action is never dithered. This is a
robustness test of the selected action under nearby continuations, not a second sample of the declared
deterministic controller. The smoke runs show that the dither moves returns only slightly (paired SE about
1e-4 in U). The re-evaluation gate is therefore close to the margin test on near-deterministic outcomes. A
stochastic actor continuation would be the alternative, but it changes rho away from H12.

## Output

Shards mirror `labor_teacher_data`: one per world, with arrays first, `meta.json` last,
`candidates.jsonl` bound by its sha256 in the meta, then `dataset.json` and `search_report.json`. Each row is
in collector format, with inputs, the full action-12 label tensors and per-head validity. A focal root gets
one row per personality, with `labor_personality` set to the personality the label is for. Extra fields:

- `row_source` (1 = search);
- `row_label_variant` (personality index, part of the shard identity; `verify_shard` now includes it when
  present);
- `search_label_is_h12`, `search_label_family`, `search_best_family`;
- `search_selection_gain`, `search_reeval_gain`, `search_reeval_se`;
- `search_horizon`, `search_candidates`, `search_label_personality`.

Label validity: a head is valid when the winning candidate's source chose it, meaning H12-valid heads, every
actor head, and the perturbed block. Heads inherited from an H12 drive default stay invalid, as in the H12
label.

`reward_*`/`transition_*` hold the label action's first transition only. **Search rows are not V
targets**, and they must not go through `discounted_returns`. Several rows share a (world, camp, turn).

`load_labor_dataset` and `pretrain_labor_actor_bc` load and fit the smoke dataset unchanged (verified on the
smoke shard).

The candidate table holds every candidate's components, utilities, horizon reached, step rewards,
re-evaluation returns, rejected candidates and the work per root. These are forced-first-action
finite-window returns for Q/critic diagnostics, kept distinct from V.

## Usage

```
python -m anthropocene_rl.autonomous_v7.labor_search \
  --normalizer-checkpoint ckpt.pt --actor-checkpoint ckpt.pt \
  --worlds 90100-90159 --source h12 --turns 16 --roots-per-world 8 \
  --candidates 10 --horizons 1,3 --keep all,3 --reeval-streams 2 \
  --personality dirichlet:1 --jobs 30 --out datasets/labor-search-r0
```

`--source actor` drives the root trajectory with the deterministic actor. H12 still labels and continues.

## Cost model (laptop, 8 players, Small world)

One native resolve costs about 8 s of wall time, including the fork and restore overhead. Each labor step
returns about 5 MB of JSON, so encoding and parsing it dominates, not simulation. A continuation decision is
two resolves (shadow heuristic plus labor step); step 0 is one. See the commit report for measured numbers and
the VM estimate.

### Local measurements (25 October 2026)

Actor: `r3-build-rl/evidence/labor-local-validation-bc-v1/checkpoint-bc-000.pt`, 8 players, `--source h12`.

| Run | World | Config | Roots | Resolves/root (search + re-eval) | s/root | s/resolve |
| --- | --- | --- | --- | --- | --- | --- |
| smoke 1 | 90100 | M=8, h 1,2, keep all,1/2, 3 streams, balanced | 2 | 36 (18 + 18) | 282-308 | 8.2 |
| smoke 2 | 90101 | M=10, h 1,3, keep all,3, 2 streams, dirichlet:1 | 2 | 46 (26 + 20) | 468-575 | 11.3 |

- Exact restore held on both worlds (`restore_checks`).
- Smoke 1: search beat H12 at 2 of 2 roots for every personality. The winners were actor candidates (one
  deterministic, one sampled), mainly through QoL-years (+5.2 discounted over 2 decisions).
- Smoke 2: 1 of 2 roots for every personality, the winner an actor sample. It avoided the settlement-guard cost
  (-1.0) that H12 incurs on a 20-person camp. The other root's best (+0.6% at selection) failed re-evaluation.
- At h=1 most single-block perturbations tie with H12 exactly, because the first transition barely depends on
  the labor choice. The h=1 round therefore prunes mostly by the keyed tie-break. Perturbations need h >= 2-3
  to separate.
