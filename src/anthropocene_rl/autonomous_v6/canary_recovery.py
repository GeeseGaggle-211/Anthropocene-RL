"""Fail-fast TRAIN probe checks and durable canary checkpoints before diagnostics."""

import hashlib
import json

from anthropocene_rl.autonomous_v4.diagnostics import probe_face


def validate_probe(obs):
    subset, labels, roles, feature = probe_face(obs)
    if float(subset["x"][:, feature].std()) <= 0:
        raise ValueError("Reversal feature is constant")
    assignments = {}
    for i in range(len(labels)):
        key = subset["x"][i].numpy().tobytes()
        if key in assignments and assignments[key] != float(labels[i]):
            raise ValueError("Identical actor input has conflicting reversal labels")
        assignments[key] = float(labels[i])
    return dict(
        rows=len(labels),
        feature=feature,
        feature_std=float(subset["x"][:, feature].std()),
        roles=roles.tolist(),
    )


def persist_rollout(trainer, output, cell, episodes):
    checkpoint = output / f"{cell['id']}-step{cell['transitions']}.pt"
    temporary = checkpoint.with_suffix(".tmp")
    trainer.checkpoint(temporary)
    temporary.replace(checkpoint)
    trace = output / f"{cell['id']}-training-traces.json"
    temporary = trace.with_suffix(".tmp")
    temporary.write_text(
        json.dumps([{k: v for k, v in e.items() if k != "final_obs"} for e in episodes]) + "\n"
    )
    temporary.replace(trace)
    cell.update(
        checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        durable_transitions=cell["transitions"],
    )
    return checkpoint
