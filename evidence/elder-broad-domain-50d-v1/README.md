# Elder broad-domain 50-decision bank handoff

Date: 2026-10-12

## Disposition

**`NATIVE_SCENARIO_CONSTRUCTION_BLOCKED` — bank not promoted, normalization not
fit, PPO not started.**

The frozen native simulator did not crash and did not exhibit replay, schema,
SaveCodec, or numerical instability. Generation was deterministic. The blocker
is narrower: under the current fixed-time F15 constructor, both members of the
mild-food-import / food-surplus-export counterfactual pair do not reliably have
their required *delivered* trade direction in the same measured interval.

## Identities

- Learner execution base: `c0628fcd2d348b709ce7abca6972390c6a130f84`
- Native simulator: `5559961164d6f9da8e6d20c1dc563ecec3eb04b7`
- Final attempted generator config SHA-256:
  `70bece532630d25a7db954beff7df5d0b1b402b204fddbaffbed6ce1bb089953`
- Bank identity: `elder-v2-broad-domain-50d-pilot-v1`
- Requested candidates: 800
- Target accepted states: 499
- Frozen partition salt: `elder-v2-broad-domain-50d-partitions-v2-830`

Movement and husbandry are admitted through native masks. Fishing is excluded.
No native economics, ecology, reward, observation/action schema, or physics was
changed.

## Full-generation result

The bounded 800-candidate pass completed natively with 697 candidate states and
103 controlled rejections:

- candidate splits: TRAIN 487, DEVELOPMENT 99, HELDOUT 111;
- rejections by family: F05 7, F06 1, F07 12, F15 82, F17 1;
- rejection classes: native generation 5, requested stratum not achieved 78,
  requested stratum unreachable 20.

The exact selector could satisfy the non-F15 quotas. It stopped because the F15
mild-import / surplus-export contrast had only three complete eligible pairs;
five were required. No partial sibling, relaxed physical tag, or replacement
partition was accepted.

## Focused F15 evidence

The 24-pair focused diagnostic found:

- three complete pairs at decision 1;
- all 27 rejected siblings had a valid observed contact but no delivered trade
  in the measured interval;
- 24/27 later showed positive contact trade flow during a legal five-step
  calibration trace;
- the same worlds at decision 2 produced two complete pairs, different from the
  decision-1 successes;
- decision 3 produced no complete pair.

The predeclared final panel then used exactly 24 new independent TRAIN groups,
three documented factor/timestamp configurations, and no outcome-based seed
selection. It produced seven accepted records but only one complete pair, below
the gate of two. An independent repeat reproduced the native result and all
seven blobs byte-for-byte. Native-result SHA-256:
`522b6373acce9e5a2a8f3b2845de1390a0aaf7379f92563353a73a58183d548d`.

## Learning and architecture conclusion

There is no new training trajectory for this generation because promotion
stopped before normalization and PPO. The prior narrow PPO campaign did **not**
show a steady increase in cultivation intensity: mean requested field operation
fell from 0.4865 to 0.3782, and requested Development budget fell from 0.5071 to
0.2993. Its 3-decision episodes and roughly 2.3k–4.6k transitions per seed were
too short and underpowered for an architecture conclusion.

The current blocker also provides no evidence against the MLP. Keep the
separate `512 -> 256 -> 256` tanh actor and critic, critic-only LayerNorm, and
normalization-v3. Do not introduce ReLU, actor LayerNorm, recurrence, a
Transformer, or a privileged critic at this boundary.

## Recommended rectification

Review `docs/F15_EVENT_ALIGNED_SCENARIO_CONSTRUCTION_PROPOSAL_2026-10-12.md`.
It proposes a new group-level capture rule: run both counterfactual siblings in
lockstep for a fixed five-decision construction horizon and capture both only at
the earliest shared decision where the exact delivered import/export predicates
hold. If no shared event occurs, reject the whole pair.

After an approved, finite v2 validation passes, freeze the bank, publish it to
GCS, fit a fresh broad-domain TRAIN-only normalization-v3 artifact, run the
headroom/representation probes, and only then begin powered 50-decision PPO.

Full native traces and snapshot blobs are external run artifacts. Git retains
only this handoff, the source/configuration, and compact hash-anchored evidence.
