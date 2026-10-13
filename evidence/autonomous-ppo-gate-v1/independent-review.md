# Independent review: autonomous parent-Elder PPO gate v1

**Verdict: GO for the proposed capped CPU parent-Elder canary, within the envelope below.** No unresolved correctness blocker was found in the reviewed native boundary, learner likelihood/update mechanics, or pilot evidence. This is a technical readiness decision for a monitored experiment; no canary was launched by this review.

Reviewed native simulation source SHA-256: `8a1319571e1204a8330306277b96529de23ec3932efff11c61d1964efe996f52` (sorted simulation filenames, NUL separators, and file bytes). The independently recomputed value matches pilot provenance. All 17 files in the retained artifact inventory matched their recorded SHA-256 values.

## Independent checks

- Reviewed the requested-share Dirichlet/Beta likelihood, masked categorical movement, conditional quote likelihood and support, inactive-head RNG behavior, frozen rollout observations/masks, PPO clipping/KL stop, physical-time reward/GAE, truncation bootstrap, and checkpoint restoration.
- Reviewed opening-stock settlement before movement, private site/contact address tokens, destination economic execution, and separation of diagnostic telemetry from actor features. The inference plug-in evaluates each actor from its own observation.
- Ran `tests/test_autonomous_v4.py` with the current native bridge and curriculum bank enabled: **8 passed**.
- Independently ran `scripts/check_autonomous_resume.py` against all three final checkpoints with populated Adam state: **PASS**. Native fingerprints, likelihoods, model and optimizer hashes, sampler/shuffle RNG, and update metrics reproduced exactly. The resulting JSON exactly matched retained evidence. These were disposable validation updates; retained checkpoints were unchanged.
- Independently ran `scripts/check_autonomous_curriculum_witnesses.py`: **4 focal moves, 1 focal settled trade, 5 focal fissions, 308 focal migrants**, with replay verification. Output exactly matched retained scripted evidence.
- Recomputed results from raw episode files and inspected initial/final checkpoints: all three seeds have 2,048 retained training transitions; actor, critic, and head parameters changed. All 96 final development episodes survived. Every family improved for every initialization. Equal-family welfare changed approximately **28.9916→90.4358**, **28.9887→119.0886**, and **29.0102→89.2022**.
- Inspected the reported full native/learner regression results, old-mode fingerprint parity, and performance evidence. This final review did not rerun the entire full regression or benchmark suites.

Earlier actionable findings were resolved: contact-slot/context alignment, stale relocated-contact settlement, Housing Material tool-capacity enforcement, finite timber pressure requests, and microscopic-stock scripted quotes. Native continuation validation now includes populated Adam state.

## Scope and required canary envelope

Retain immutable pilot finals. Permit at most **32,768 additional focal transitions per seed, 98,304 total**, CPU only, at most four native workers, one Torch thread, and **four hours maximum**. Evaluate every 4,096 transitions using the frozen paired development protocol. Abort on any illegal action, nonfinite result/gradient, or replay failure; survival below 75%; welfare below 80% of the corresponding pilot final; or unexplained memory/throughput deterioration exceeding 2×. Preserve all new checkpoints separately.

The present runner deliberately supports only the small pilot. The separate canary orchestration must enforce these limits and pass worker-partition/replay and checkpoint-equivalence preflight before concurrency is raised. This review does not certify an unimplemented parallel collector. A changed training-only market curriculum must have its own version and preserve the frozen development suite.

## Limits on the GO decision

This pilot demonstrates learning and functioning mechanics on a small cultivation-based parent-Elder curriculum. It does not demonstrate robust village sustainability, learned trade competence, husbandry/fishing breadth, or dynastic control. Only two learned training quotes settled across all seeds, and none settled in final development. Scripted trade witnesses establish settlement mechanics, not learned trading ability. Parent retention and food coverage remain weak in some families; fission and migration also change the parent population.

The GO therefore excludes broad production scale-out, GPU training, and learned daughter/dynastic self-play. Repeated learned trade settlement on unseen worlds, broader shortages/capabilities, longer horizons, and a separate daughter/reward design remain promotion gates. No acceptance threshold was waived.
