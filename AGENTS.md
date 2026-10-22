# Anthropocene learner working agreements

## Training workload sampling

- Use the local machine only for validation and stability probes, with at most three seeds for any training workload.
  Local results are diagnostic and may not select an arm, tune a treatment, or establish readiness.
- Run readiness-bearing training comparisons on the VM at no smaller sampling scale than the T3 experiment: three
  matched arms and five optimizer seeds per arm (15 runs), with the same seed identities paired across arms. A different
  arm/seed design requires a reviewed pre-registration; a convenient smaller sample is not a substitute.
- Keep confirmation seeds 3000-3059 sealed until an explicitly authorized milestone. VM scale prevents the local
  three-seed stability sample from becoming a biased selection screen.

Current reviewed Round-3 details and seed roles are normative in `docs/MODEL9_ROUND3_BUILD_ARCHITECTURE.md` and
`docs/MODEL9_M7.md`.
