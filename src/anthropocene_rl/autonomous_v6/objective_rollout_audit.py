"""Versioned objective PPO rows, including immutable own reset/value coordinates."""

import copy
import hashlib
from pathlib import Path

import torch

from anthropocene_rl.autonomous_v4.policy import Action


@torch.no_grad()
def save_rollout_audit(trainer, rows, path, *, rollout_provenance):
    """Save the actual retained rollout; never recollect or substitute value targets.

    Call after checkpointing a completed update and before optional probes. Values
    and likelihoods are rescored on that rollout's own observations and latent
    requests. No policy sampling or optimizer step occurs. These in-sample GAE
    diagnostics are distinct from fresh world-disjoint critic validation.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if not rows or trainer.config.get("model_version") != "autonomous-ppo-model-v8":
        raise ValueError("A nonempty model8 rollout is required")
    saved = []
    numeric = ["value", "next_value", "elapsed", "reward", "advantage", "return"]
    for row in rows:
        record = {
            "obs": {k: v.detach().cpu().clone() for k, v in row["obs"].items()},
            "action": {
                k: getattr(row["action"], k).detach().cpu().clone()
                for k in ["workforce", "activities", "controls", "movement", "trade"]
            },
            "old_logp": row["logp"].detach().cpu().clone(),
            **{k: float(row[k]) for k in numeric},
            **{k: bool(row[k]) for k in ["terminal", "truncated"]},
        }
        if not torch.isfinite(torch.tensor([record[k] for k in numeric])).all():
            raise ValueError("Nonfinite rollout target or physical interval")
        if record["elapsed"] <= 0 or (record["terminal"] and record["truncated"]):
            raise ValueError("Invalid physical interval or termination flags")
        if not torch.isfinite(record["old_logp"]).all():
            raise ValueError("Nonfinite rollout likelihood")
        saved.append(record)
    observations = {k: torch.cat([r["obs"][k] for r in saved]) for k in saved[0]["obs"]}
    actions = Action.cat([Action(**r["action"]) for r in saved])
    _, post_logp, _, values = trainer.model(observations, actions)
    if not torch.isfinite(post_logp).all() or not torch.isfinite(values).all():
        raise ValueError("Nonfinite post-update likelihood or critic value")
    returns = torch.tensor([r["return"] for r in saved], dtype=values.dtype)
    error = values.cpu() - returns
    variance = float(returns.var(unbiased=False))
    summary = dict(
        rows=len(rows),
        target_variance=variance,
        mse=float(error.square().mean()),
        mean_value_bias=float(error.mean()),
        explained_variance=(1 - float(error.var(unbiased=False)) / variance)
        if variance > 1e-8
        else None,
        mse_over_return_variance=float(error.square().mean()) / variance
        if variance > 1e-8
        else None,
        label="Current retained rollout GAE targets, in-sample; not independent critic readiness",
    )
    temporary = path.with_suffix(".tmp")
    torch.save(
        dict(
            version="model8-native6-objective-rollout-audit-v1",
            config=copy.deepcopy(trainer.config),
            checkpoint_provenance=copy.deepcopy(trainer.provenance),
            rollout_provenance=copy.deepcopy(rollout_provenance),
            counters=dict(
                episodes=trainer.episodes, transitions=trainer.transitions, updates=trainer.updates
            ),
            rows=saved,
            post_update_values=values.detach().cpu().clone(),
            post_update_logp=post_logp.detach().cpu().clone(),
            metrics=summary,
        ),
        temporary,
    )
    temporary.replace(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **summary)
