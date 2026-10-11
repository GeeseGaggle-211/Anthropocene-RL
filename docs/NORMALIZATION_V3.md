# Elder normalization-v3

Normalization-v3 is a frozen, TRAIN-only semantic transform for the Elder v2
512-feature observation contract. It is backward-compatible at load time with
normalization-v2; checkpoints retain the normalizer version and complete fit
metadata, so inference never changes running statistics.

## Fit contract

- Fit inputs are PILOT TRAIN reset observations plus five short legal native
  calibration steps per accepted TRAIN snapshot.
- Every scenario family contributes equal total fit weight. DEVELOPMENT and
  HELDOUT are transform-only diagnostics.
- Binary and bounded features retain their existing exact mappings.
- Positive and signed dense channels use the larger of the family-weighted
  95th percentile and the unweighted 99.5th-percentile tail envelope.
- Sparse positive and signed channels additionally use the 99th percentile of
  their nonzero TRAIN values.
- Repeated sparse migration and trade channels share semantic scales across
  contact slots; commodity import/export event channels are pooled likewise.
- Positive ratios use a finite TRAIN-fit softening constant. A physical zero
  remains distinct from missingness and cannot create an infinite log value.
- The fitted scale, fit mode, nonzero count, ratio softening, weighting rule,
  source mix, manifest identity, and native/schema identities are serialized.

## Promotion gates

The TRAIN calibration set must contain at least 32 nonzero observations for
each declared sparse causal group. Signed migration groups must additionally
contain at least 16 positive and 16 negative values. Per-feature saturation is
reported for TRAIN, DEVELOPMENT, HELDOUT, and the declared physical slices.

The final remediation bank passes every sparse-support gate and has no feature
above 10% saturation in TRAIN or any reported causal TRAIN slice. DEVELOPMENT
peaks below 2%; HELDOUT has no observed saturation at the 0.95 absolute-output
threshold.

## Network decision

The actor and critic remain separate `512 -> 256 -> 256` tanh MLPs. LayerNorm,
ReLU/SiLU, recurrence, and a privileged critic are not part of normalization-v3.
The pilot records preactivation and hidden-activation distributions for both
networks. A LayerNorm/activation ablation is recommended only if at least 10%
of hidden activations remain at `|tanh(h)| >= 0.95` after the input transform is
fixed, or if another clear hidden-conditioning defect appears.
