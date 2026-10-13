# Critic step attribution v1

The unchanged continuation v4 gate failed at seed29/step20480. Recipe tasks4/9
share the exact critic observations, labels and error trajectory. Both actor
reversals pass. Critic MSE first falls to0.16875, then oscillates and ends at
1.39740, above the frozen0.9 threshold. The independent reviewer classifies this
as update instability, not demonstrated loss of learnability. The original
failure remains authoritative; no threshold is waived.

Run one discarded critic-only attribution with retained parameters, all Adam
moments, epsilon, clipping, shuffle state and target construction. Compare the
preregistered learning rates0.0003,0.0001,0.00003, each64 updates ×4 minibatches
of32 observations. Require exact reproduction of all64 original MSE values at
0.0003 before interpreting the alternatives. The actor and critic are disjoint;
the actor is not updated and no actions are drawn. Record pre/post residuals,
gradients, parameter and prediction step norms, and gradient/step cosine.

The configuration pins checkpoint, source, probe and original diagnostic hashes.
One Torch thread,60-second deadline,768 critic minibatches maximum; zero native
steps, zero native PPO updates, zero new policy samples. Never save or promote
these replicas or overwrite prior evidence. This is a separately bounded
diagnosis after the native98304-transition envelope was exhausted. A smaller
learning rate succeeding on this task only identifies a candidate remedy; it
does not pass the original gate or authorize scale-out. Full retained-state
coverage and a separately reviewed native budget would still be required.
