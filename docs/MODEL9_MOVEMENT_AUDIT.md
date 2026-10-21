# Model9 movement audit: S-s55 learned movement

Date: 21 October 2026. Branch `feature/audit-movement` (RL repo). Simulator source: `feature/elder-model9-native`
at `f06e3be`, unchanged.

**Verdict: mixed.** About four fifths of S-s55's welfare gain is physical. It produces more food and more people,
because moving spreads the population over more footprints and onto the coast. Two simulator rules inflate the rest
and shape *how* the policy moves:

1. **Expectation reset.** A move of one tile resets the settled-expectation clock. This makes missing clothing and
   housing count 60% less in QoL.
2. **Free shuttling.** Fixed capital left behind is neither lost nor charged, and a move has no travel cost. The
   policy therefore shuttles between two or three of its own housed sites.

Fresh-tile stock mining and regrowth calibration are **not** involved.

## Setup

- Worlds: the default seeds 2000-2019 and 2026-2059 (54 worlds), 25 turns, withheld 0. The fishing capability fixture
  is seeds 2000-2009 with `contracts.CAPABILITY_MASKS["fishing"]`.
- Policies, paired on the same openings:
  - `s55-grouped`: S-s55 with the hierarchical region choice, the configuration under review.
  - `s55-argmax`: S-s55 with the plain argmax region choice.
  - `ref-native`: the frozen reference with heuristic movement.
  - `s55-native`, a control: S-s55's levers with heuristic movement.
  - `s55-stay`, a control: S-s55's levers with the region slot forced to the current tile, so it never moves.
- Instrumentation is a new read-only bridge op, `inspect`, in `integration/Model9BridgeProgram.cs`. For every camp it
  reports:
  - tile, elevation and village/cultivation state;
  - tenure (`StationaryTurns`, `StationaryYears`) and `RudimentaryExpectation`;
  - QoL parts (food, water, cloth, house, gear factor at the actual E and at E = 1, cohesion);
  - fixed housing at the tile and elsewhere, light housing, operated field tiles and capital, and stores;
  - the last turn's `DemographicReport` (food by source, births, deaths, split);
  - land and aquatic footprint stock and season-mean capacity, for its tile and all six neighbours, through
    `LazyEcology.ProjectEcology` only.

  The op takes `SaveCodec.Fingerprint` before and after and throws if they differ.
  `tests/test_movement_audit_inspect.py` checks that rollouts with and without `inspect` produce identical post-step
  fingerprints and replay.
- Tools:
  - `movement_audit.py` runs the rollouts.
  - `movement_audit_summary.py` produces the tables.
  - `movement_audit_forks.py` runs the paired fork tests. MOVE and STAY branches start from a byte-identical fork and
    take identical actions except the movement code.
- Definitions:
  - **Directive:** a learned keep-intent or a region row other than the current tile. This is what the 0.967 counts.
  - **Physical move:** the camp's tile changed between two consecutive inspects. Every per-move metric below uses
    physical moves.
  - **E-neutral welfare:** welfare with each camp-turn's QoL recomputed at E = 1, i.e. QoL × gearSettled / gear. It
    removes the expectation-accounting effect, not its behavioural consequences.

## 1. Moves

| metric (54 default worlds) | s55-grouped | s55-argmax | s55-native | s55-stay | ref-native |
|---|---|---|---|---|---|
| welfare / ref | **1.166** | 1.017 | 0.959 | 0.833 | 1 |
| E-neutral welfare / ref | **1.135** | 1.000 | 0.958 | 0.838 | 1 |
| residents per world | 21,974 | 18,238 | 16,284 | 14,978 | 17,318 |
| settlements per world | 79.2 | 52.5 | 40.5 | 41.3 | 39.4 |
| settlements below 30 | 3.5% | 1.3% | 0.5% | 1.0% | 0.7% |
| final QoL | 0.889 | 0.866 | 0.854 | 0.863 | 0.841 |
| final QoL at E = 1 | 0.879 | 0.855 | 0.854 | 0.863 | 0.840 |
| **directive share** (of decisions) | **0.967** | 0.286 | – | – | – |
| **physical-move share** (of camp-turns) | **0.323** | 0.198 | 0.042 | 0 | 0.041 |
| physical-move share, settled camps | 0.301 | 0.191 | 0.033 | 0 | 0.033 |
| physical-move share, nomadic camps | 0.563 | 0.271 | 0.152 | 0 | 0.140 |
| **turns per completed relocation** (mean; median) | **1.28; 1** | 1.25; 1 | – | – | – |
| distance per move | 1 tile (100%) | 1 | 1 | – | 1 |
| moves back to one of the last two tiles | **48.9%** | 62.4% | 14.2% | – | 14.4% |

- "Settled" means a village, fixed housing at the tile, or operated fields: 91-92% of camp-turns in every policy.
- Of S-s55's 60,731 "go" directives, only 20,240 (33%) relocated the camp. The other 40,491 left no intent in memory,
  meaning the request was blocked at placement or repaired. The bridge does not distinguish the two.
- S-s55 never used keep-intent. A relocation is a fresh one-step request each turn: 10,459 completed and 40,520
  abandoned or replaced.
- **So the 97% is mostly ineffective directives.** The camp actually moves on a third of its turns, one tile at a time,
  and half of those moves return to a tile it has just left.

## 2. Capital

Per physical move, measured in the state before the move:

| | s55-grouped | ref-native |
|---|---|---|
| moves from a settled camp | 85% | 74% |
| village before the move | 2.0% | 0% |
| fixed housing at the origin (units) | 27.2 | 1.1 |
| of which still standing at the origin after the turn | 24.5 | 1.0 |
| own fixed housing already waiting at the destination | **15.6 (34% of moves)** | 0.2 (1.6%) |
| fixed housing at the destination after the turn | 31.4 | 5.6 |
| operated field tiles / cultivation capital | 3.4 / 0.37 | 2.8 / 0.12 |
| camp stores before → after | 1,466 → 1,468 | 1,781 → 2,038 |
| expectation E before → after | 0.46 → 0.40 | 0.64 → 0.40 |

What a move does in the simulator, with file and line citations:

**The move has no food, labour or time cost.**
- `PreparedTurn` resolves placement before the economy (`TurnPlanning.cs:326-329`).
- `TurnResolver.Resolve` runs `Move` before `ResolveEcology` (`TurnResolver.cs:156-166`).
- The camp therefore harvests the full 20-year horizon at the destination.
- Contract-10 movement is one tile per turn along the known path (`ElderMovement.cs:7-9`; `ElderModel9.cs:627-651`).
- Stores and portable goods are camp inventory and travel with it: the measured stores are unchanged across a move.

**Fixed housing is neither destroyed nor charged.**
- It is a `FixedHousingStock` keyed by (camp, tile) that stays at its site (`TurnResolver.cs:417-420`).
- It counts only while the camp stands on that tile (`NeolithicProduction.cs:386-394, 416-420`).
- It decays at `HousingDecayPerTurn = 0.1` per reference turn whether or not it is occupied
  (`NeolithicProduction.cs:371-383, 395-398`).
- So leaving costs its *use* for the turns away, never the capital. On return it counts again.

**Fields.**
- A village that moves releases only its old local fields; the capital stays on the map and decays
  (`TurnResolver.cs:426-440`).
- This turn's harvest is capped at bridge-level intensity (`GenerationalEconomy.cs:597-611`).
- A non-village camp's field footprint is simply radius 1 around its current tile (`CultivationEconomy.cs:227`).
- Fishery sites release their operator (`TurnResolver.cs:441-451`).

**Tenure resets, and that lowers expectations.**
- A move sets `StationaryTurns = 0` and, for non-villages, `StationaryYears = 0` (`TurnResolver.cs:452-455`).
- The planning view does the same (`TurnPlanning.cs:724-731`).
- `RudimentaryExpectation = 0.25 + 0.75 · min(1, StationaryYears / 100)` (`NeolithicProduction.cs:1120-1121`).
- The clothing and housing QoL terms are `1 − E · (1 − coverage)` (`NeolithicProduction.cs:1122, 1144-1148`).
- After one move E drops from up to 1.0 to 0.40, so a clothing or housing shortfall costs 60% less QoL.
- The planners value clothing and housing with the same E (`ProductionDemand.cs:597-605, 855-862`), so they also buy
  less of them.
- The only other cost of moving is the hinterland-knowledge radius reset (`ContactKnowledge.cs:207`).

## 3. Fresh tiles

Food per resident per turn, from the `DemographicReport` after the turn:

| | arrival turn (moved) | tenure 2 | tenure ≥ 3 |
|---|---|---|---|
| s55-grouped: land / fish / herd / total | 64.5 / 20.9 / 3.9 / **89.9** | 81.6 / 14.6 / 1.3 / 97.8 | 70.0 / 12.6 / 1.1 / **83.9** |
| ref-native: land / fish / herd / total | 86.3 / 12.5 / 1.7 / 100.7 | 99.8 / 10.0 / 0.9 / 110.8 | 74.6 / 9.4 / 1.2 / 85.2 |

Land footprint stock relative to its season-mean capacity:

| | s55-grouped | ref-native |
|---|---|---|
| destination before arrival | 0.754 | 0.797 |
| origin before departure | 0.754 | 0.772 |
| destination / origin land stock (median) | 0.97 | 1.22 |
| steady state under occupation (≥ 3 turns) | 0.785 | 0.825 |
| abandoned footprint, 0 → 1 → 3 → 6 turns after leaving | 0.749 → 0.758 → 0.771 → 0.777 | 0.745 → 0.773 → 0.781 → 0.787 |

- **There is no fresh-tile windfall.** Occupied footprints sit at 75-83% of capacity, and S-s55 moves onto footprints
  no fuller than the one it leaves.
- Land food per resident on arrival is *lower* than at incumbent sites.
- Land harvest is a small share of standing stock: land availability is the opening stock
  (`LazyEcology.cs:532-537`), taken at `GatherFraction` 0.22 (`GenerationalEconomy.cs:613-615`).
- Abandoned land therefore barely needs to regrow. Regrowth calibration is not the lever.
- The higher food per mover comes from fish.

## 4. After arrival

Changes from the turn before the move. Movers are compared with non-movers in the same world, turn and log2
population bucket that held their tile for the two previous turns.

| s55-grouped | arrival (k=0) | k=1 | k=3 |
|---|---|---|---|
| ΔQoL, mover vs control | +0.017 vs +0.006 | +0.030 vs +0.020 | +0.057 vs +0.056 |
| ΔQoL at E = 1, mover vs control | **+0.015 vs +0.025** | +0.054 vs +0.045 | +0.118 vs +0.088 |
| Δpopulation, mover vs control | +18.3% vs +12.9% | +35.3% vs +24.3% | +64.2% vs +50.0% |

- The QoL a mover reports on arrival exceeds the control's only because of the expectation reset. At E = 1 the mover
  is behind on arrival.

**Paired fork test (216 pairs).** At turns 4, 8, 12 and 16, MOVE and STAY run for four turns with identical levers.

| 4-turn horizon | welfare | E-neutral welfare | residents | fissions | food produced | cloth | house | E |
|---|---|---|---|---|---|---|---|---|
| MOVE / STAY | **0.997** | **0.967** | 1.016 | **1.75** | 0.992 | 0.909 vs 0.955 | 0.798 vs 0.822 | 0.72 vs 1.00 |

- **Over a four-turn horizon, the policy's marginal moves gain nothing.** Reported welfare is flat because the
  expectation reset hides a real fall in clothing and housing. E-neutral welfare is 3.3% lower, and fissions nearly
  double.

**Full-episode paired fork (54 worlds).** MOVE and STAY start from turn 0 and run all 25 turns.

| MOVE / STAY | welfare | E-neutral | residents | settlements | fissions | food | land food | fish food |
|---|---|---|---|---|---|---|---|---|
| ratio | **1.40** | **1.35** | 1.47 | 79.2 vs 41.2 | 70.6 vs 32.2 | 1.36 | 1.25 | 2.17 |

- The welfare ratio over the heuristic-movement control rises with time:

  | turns | raw ratio | E-neutral ratio |
  |---|---|---|
  | 0-1 | 1.60-1.70 | 1.12-1.13 |
  | 4-10 | 0.85-0.96 | 0.73-0.91 |
  | 24 | 1.40 | 1.38 |

- The early "gain" is expectation. The late gain is population.

## 5. Settlement growth

| | s55-grouped | s55-argmax | s55-native | s55-stay | ref-native |
|---|---|---|---|---|---|
| fissions per world | **70.6** | 43.6 | 31.6 | 32.3 | 30.5 |
| fissions per founding lineage | **9.6** | 6.3 | 4.9 | 5.1 | 4.8 |
| share in a turn the parent physically moved | **71%** | 43% | 15% | 0 | 15% |
| mean daughter size | 69 | 91 | 96 | 99 | 99 |
| fission triggers (Distress / Crowding) | 3,263 / 550 | 2,011 / 344 | 1,448 / 257 | 1,642 / 104 | 1,393 / 255 |

- **Yes: the doubling of settlements comes from fission after moves.**
- A moved camp splits on 13.3% of its move-turns, against 2.6% for stayers. The reference shows the same pattern,
  15.9% against 4.0%.
- The arriving parent has left its fixed housing behind: housing coverage is 0.60 against 0.86 for stayers. This
  lowers its baseline in the distress and crowding fission tests (`Fission.cs:738-757, 143-210`).
- More, smaller settlements then cover more footprints (+25% land food) and carry lower crowding and disease hazards,
  which scale with settlement headcount (`Demography.cs:118-125`).
- Deaths per person-turn are 0.449 against 0.554 for the reference; births are 0.533 against 0.628.
- Immigration is unchanged, at 55 against 57 flows per world.

## 6. Fishing

On the fishing fixture (10 worlds, cultivation and husbandry withheld):

| | s55-grouped | s55-stay | ref-native |
|---|---|---|---|
| welfare / ref | 1.189 | 0.854 | 1 |
| E-neutral welfare / ref | **1.052** | 0.873 | 1 |
| final QoL (at E = 1) | **0.664 (0.632)** | 0.748 | 0.762 (0.755) |
| episode clothing / housing coverage | **0.587 / 0.498** | 0.736 / 0.591 | 0.731 / 0.613 |
| coastal population share | 0.54 | 0.33 | 0.40 |
| aquatic food share | 0.27 | 0.17 | 0.19 |
| settlements | 79.4 | 37.2 | 38.4 |

- **Camps are not leaving the shallows.** They are more coastal, and fish more, than under the reference.
- Fishing-world QoL falls because real clothing and housing coverage collapse among shuttling, fissioning camps
  without crop or animal fabric. Small settlements make up 46% of camp-turns.
- The expectation reset hides most of that collapse from QoL, and therefore from the reward: 13.7 of the 18.9 welfare
  points over the reference disappear at E = 1.

## Mechanism and attribution

The default-world gain over the reference is 16.6 points:

| source | points | share |
|---|---|---|
| **Expectation-reset artefact** | ≈ 3.1 | ≈ 19% |
| QoL gain at E = 1 (cohesion of smaller settlements) | ≈ 1.5 | |
| More person-years (+11.7%), from more fission, more footprint and coastal fishing | ≈ 12 | |

- The expectation-reset effect comes from `TurnResolver.cs:455`, `TurnPlanning.cs:729-730` and
  `NeolithicProduction.cs:1120-1148`. It covers nearly all of the early-episode spike.
- The population gain is lawful under current rules. But it is driven by moves that are free and keep capital, and by
  the reset that makes leaving housing cheap in QoL terms. That encourages the 1-tile A↔B shuttle: 49% of moves
  return, and 34% land on the camp's own housing.

## Proposed minimal simulator fix

Separate **settled expectation**, a property of the people, from **site tenure**, a property of the tile.

1. **Change the move to preserve expectation.** In `TurnResolver.cs:452-455`, keep `StationaryTurns = 0` (site
   tenure) but stop zeroing `StationaryYears` on a move. Do the same in the planning copy at
   `TurnPlanning.cs:724-731`. A softer variant decays the clock instead of zeroing it, for example by subtracting one
   turn's worth of years per move.
2. **Re-point shelter routing.** The shelter-routing test `demonstratedFixedTenure` reads `StationaryYears`
   (`DemandControl.cs:770-781`). Point it at `StationaryTurns` × turn years, or at a new optional `SiteYears` field, so
   portable-versus-fixed routing keeps its meaning. `SaveCodec.cs:153` already derives years from turns for old saves.
3. **No other changes are needed.** `Fission.cs:152, 310`, `ProductionDemand.cs:604, 861` and
   `SocialRelocation.cs:815` read `RudimentaryExpectation` and follow automatically.

Expected effect:
- The reported QoL spike on moving disappears.
- Shuttling to own housing loses its point.
- Learned movement is then judged on its physical gains, which the full-episode fork puts at about +35% over never
  moving and +18% over heuristic movement in E-neutral welfare.

A move cost, such as a one-turn harvest penalty for distance, is a second-order option: re-measure after fix 1 before
adding it. Regrowth calibration needs no change.

## Confidence and limits

- **High confidence** in:
  - the expectation mechanism: exact formula, measured E = 0.40 after moves, and the paired forks;
  - no fresh-stock windfall: direct pure stock reads on 20,000 moves;
  - free housing retention: both code and measurement.
- **Medium confidence** in the attribution of the population gain to fission and expansion. It is correlational
  within rollouts, but the full-episode fork is causal for "movement versus none".
- E-neutral welfare removes the expectation's *accounting* effect only. The planners' reduced clothing and housing
  production under low E is a behavioural channel that only the simulator fix itself can remove. Re-run this audit
  after the fix.
- The fishing-fixture sample is small (10 worlds).
- S-s55 was trained under the current rule. A fixed rule needs retraining before the policy itself can be judged.
- The 66% of "go" directives that move nothing were not split into placement blocks and compiler repairs.
