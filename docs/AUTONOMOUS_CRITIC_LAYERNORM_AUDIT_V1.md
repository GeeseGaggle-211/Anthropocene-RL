# Critic LayerNorm conversion audit v1

The learning-rate attribution establishes a remedy for the observed critic
oscillation. It does not show that longer training will avoid representation
saturation. Existing critic tanh saturation reaches 66–78% on the fixed TRAIN
probe; smaller optimizer steps could merely delay deterioration.

Before any new native training, audit all six technically selected continuation
checkpoints on that same 192-observation TRAIN-only probe. Insert sample-wise
LayerNorm immediately before each critic tanh, epsilon 1e-5, affine scale one and
bias zero. Preserve every existing parameter by name and value; the actor and
all action heads remain exact. Compare values and representation statistics
against the unchanged model. Changing only critic LR leaves entry values exact.

Report value mean/RMS/maximum disruption, value correlation and pairwise order
preservation, saturation (absolute tanh >= .99), mean tanh derivative, centered
activation effective rank, and input-gradient norms. Verify original Adam
tensors and checkpoint bytes remain unchanged. No normalization is fitted to
evaluation data. No converted model is saved or promoted. Freeze script, probe,
model-source and checkpoint hashes before launch. Cap at six checkpoints, one
Torch thread, 60 seconds, zero actions, optimizer updates or native steps.

This audit has no scale-out pass threshold: lower saturation is an expected
mechanical effect, not proof of better values or continued learning. A subsequent
discarded adaptation comparison must use identical original-critic value-plus-
label targets, retained named Adam moments, LR .0001 for both candidates, and
new optimizer state only for the added LayerNorm parameters. Re-centering each
candidate's targets would hide conversion damage and is not acceptable evidence
of preserving useful values. Further native training needs a new bounded gate;
the previous 98,304-transition continuation envelope is exhausted. Initially keep
eight-turn curriculum and reward horizon fixed to isolate the optimizer versus
architecture effects before considering 25-turn training.
