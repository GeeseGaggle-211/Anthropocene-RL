# Proposed staged scale-out for autonomous Elder model v3

Independent final review gives GO for the capped CPU continuation described below within cultivation/parent-Elder scope. It has not been launched. This plan does not authorize broad or expensive PPO, GPU work, or the later curriculum/worker expansions without their separate gates.

## First expansion: finish the existing CPU envelope

After the remedial 8,192-transition gate passes, the cumulative additional training count is 49,152: seed 17=32,768 and 29/43=8,192 each. The original 98,304-transition envelope therefore permits at most 24,576 further transitions each for 29/43, total 49,152; seed 17 has no allowance. Use the final v3 checkpoints with their actual Adam/RNG states, same pinned simulator and frozen bank, four CPU workers and one Torch thread. Do not restart from a selected development-best checkpoint.

Preserve the cumulative four-hour native campaign cap. The completed v2/v3 campaigns used 1906.766 seconds, leaving at most 12493.234 seconds, including evaluations in any continuation. Do not reset that allowance on resume.

Checkpoint/evaluate every 4,096 transitions. Report QoL×population×time, survival, food/material/tool shortfalls, requested versus executed staffing, quote exposure and settlement, movement, migration and fission. Economic decline is a result to analyze, not a mechanical collapse veto. Preserve the fixed fresh-gradient probe and actual optimizer/resource/replay monitors. Select, per seed, the union of (a) final checkpoint, (b) checkpoint with the maximum single actor-or-critic-layer saturation fraction, and (c) checkpoint with maximum legal nondegenerate categorical confidence over the fixed probe. Break ties by earliest native transition count and deduplicate. Run retained-Adam 13-head reversals on that exact set. The next gate's separate discarded diagnostic cap is six checkpoints  × 13 tasks  × 64 updates  × 32 draws =159,744 draws, with a 900-second cap per checkpoint and no silent extension. This is prospective and separate from the completed gate's 200,000-draw cap. Any technical failure pauses the relevant seed for attribution.

## Broaden one dimension at a time

1. Add held-out world/resource/stock configurations and longer native episodes. Keep all four curriculum families and all demographic/trade/movement mechanisms active. Include changing resource availability and exchange opportunities, so course reversal matters. Preserve the native objective; diagnostic synthetic rewards never enter retained training.
2. Test learned reciprocal peers and multi-Elder WEGO collection. Keep known-counterparty observations isolated and canonical submission order. Measure compatible-quote exposure separately from settlements. Zero learned settlements is an economic/curriculum limitation, not proof of dead gradients; it still prevents a claim of trade competence.
3. Add advanced-era and rare recipe/livestock/capital legal faces with explicit coverage before training them at scale. The current13-task diagnostic covers the active cultivation faces, not every theoretical head or future technology. Fission daughters currently retain heuristic control; daughter credit assignment and inherited learned policies require a separate contract/gate.
4. Increase native worker count only after exact1-versus-N replay/rollout/update equivalence and complete-world throughput/RSS measurements. The collector currently supports1/2/4 workers. Do not assume more workers or a GPU will improve native simulator throughput. Keep actor/critic and action-distribution versions explicit; do not overwrite the five-head model.

At each stage require finite probabilities/likelihoods, legal masks, conservation/replay and actual retained-optimizer responsiveness. Report objective failures without rewriting them as technical collapse. Actor LayerNorm does not guarantee adaptable output logits; retain the exact legal mixture and prospective reverse-KL monitoring. A rising saturation statistic prompts investigation, while a failed controlled adaptation test blocks promotion.

No main merge, checkpoint overwrite, GPU ecology, or expensive scale-out is part of this gate.
