# Five-decision structural and delayed-credit audit

This zero-update audit tests whether the frozen cultivation cohort remains
scope-safe through five decisions and whether Development effects that are flat
through decision three become physically visible by decision five.

## Identities

- Audit learner commit: `4141b9ae2e9a7f7f5ddc483084560db42ca41451`
- Training learner commit: `0853ac2f6e8a239199352911c60f2249e3014e7d`
- Native simulator commit:
  `5559961164d6f9da8e6d20c1dc563ecec3eb04b7`
- Snapshot manifest SHA-256:
  `1cfabbc6057c063d2bca8d27c4bda861a090d5811e2b02f3fe51adc9fd2a6b83`
- Normalizer SHA-256:
  `6563cea5d26a54f5914b666c28584066eb18a681e5fad0559a10207a78fa01c4`
- Prior action-credit evidence SHA-256:
  `c81dbea2490e1efff0267b5eacde1ef1796489c879d296ea1a8cab3997d9c3be`
- This evidence SHA-256:
  `f492ae60f3fe081a11f4e1a9075698cc6c6e6b20ebe2f68d931943993b3ac1c7`
- Scope: TRAIN only; DEVELOPMENT and HELDOUT were not read

## Result

Status: `DELAY_HYPOTHESIS_NOT_SUPPORTED`.

The structural matrix passed all 210 TRAIN snapshots under nine constant
field/Development pairs, two alternating extremes, and the three selected
policies: 2,940 episodes and 14,700 native transitions. It recorded zero scope,
replay, requested-action, finite-value, or reward-denominator failures.

| Seed | Eligible F03 probes | Informative probes / snapshots | Delayed responsive | 3-step responsive | 5-step responsive | Gain | Gate |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1701 | 24 | 12 / 7 | 0.333 | 0.000 | 0.167 | 0.167 | FAIL |
| 1901 | 24 | 15 / 8 | 0.200 | 0.292 | 0.417 | 0.125 | FAIL |
| 2101 | 24 | 13 / 8 | 0.231 | 0.083 | 0.208 | 0.125 | FAIL |

Every seed met the informative-cohort gate. No seed met both the predeclared
minimum delayed-effect fraction of 0.50 and responsive-rate gain of 0.15.
Therefore a simple three-to-five-decision PPO ablation is not justified. The
result does not answer whether the normative 20--50-decision task has useful
long-horizon headroom; that requires the no-PPO oracle headroom audit.

`summary.json` is the compact canonical record. The raw per-seed trajectories,
cohort file, and 35 MB structural trace are durable run artifacts and must be
published to the immutable GCS run store rather than committed to Git.
