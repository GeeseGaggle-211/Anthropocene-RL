import copy

import pytest
import torch
from test_autonomous_readiness_v7 import batch, config

from anthropocene_rl.autonomous_v4.policy import Action
from anthropocene_rl.autonomous_v5.causal import protected_state
from anthropocene_rl.autonomous_v6.rollout_audit import save_rollout_audit
from anthropocene_rl.training.autonomous_ppo_v7 import AutonomousTrainerV7


def rollout(trainer):
    x = batch()
    with torch.no_grad():
        action, lp, _, value = trainer.model(x, generator=trainer.sampler)
    return [
        dict(
            obs={k: v[i : i + 1] for k, v in x.items()},
            action=action.index(torch.tensor([i])),
            logp=lp[i : i + 1],
            value=float(value[i]),
            next_value=float(value[i]) + 0.5,
            elapsed=20,
            reward=0.4,
            terminal=False,
            truncated=i == 7,
            advantage=float(i - 3),
            **{"return": float(value[i]) + i * 10},
        )
        for i in range(8)
    ]


def test_audit_retains_actual_rollout_scores_and_values_without_sampling_or_updates(tmp_path):
    torch.set_num_threads(1)
    tr = AutonomousTrainerV7(dict(config(), epochs=1, minibatch=4), 19, {}, width=16)
    rows = rollout(tr)
    tr.update(rows)
    before = copy.deepcopy(tr.model.state_dict())
    protected = protected_state(tr)
    path = tmp_path / "audit.pt"
    result = save_rollout_audit(tr, rows, path, rollout_provenance={"world": "fitting-only"})
    saved = torch.load(path, weights_only=False)
    observations = {k: torch.cat([r["obs"][k] for r in saved["rows"]]) for k in rows[0]["obs"]}
    actions = Action.cat([Action(**r["action"]) for r in saved["rows"]])
    with torch.no_grad():
        _, lp, _, values = tr.model(observations, actions)
    assert torch.equal(lp, saved["post_update_logp"])
    assert torch.equal(values, saved["post_update_values"])
    for original, stored in zip(rows, saved["rows"], strict=True):
        assert torch.equal(original["logp"], stored["old_logp"])
        assert stored["advantage"] == original["advantage"]
        assert stored["return"] == original["return"]
        assert stored["next_value"] == original["next_value"]
        assert all(torch.equal(v, stored["obs"][k]) for k, v in original["obs"].items())
    assert protected_state(tr) == protected
    assert all(torch.equal(v, tr.model.state_dict()[k]) for k, v in before.items())
    assert result["rows"] == 8 and len(result["sha256"]) == 64
    rows[0]["obs"]["x"].fill_(100)
    assert not torch.equal(saved["rows"][0]["obs"]["x"], rows[0]["obs"]["x"])
    with pytest.raises(FileExistsError):
        save_rollout_audit(tr, rows, path, rollout_provenance={})


def test_audit_rejects_nonfinite_targets_and_labels_constant_return_metrics(tmp_path):
    tr = AutonomousTrainerV7(config(), 19, {}, width=16)
    rows = rollout(tr)
    for row in rows:
        row["return"] = 1.0
    result = save_rollout_audit(tr, rows, tmp_path / "constant.pt", rollout_provenance={})
    assert result["explained_variance"] is None
    assert result["mse_over_return_variance"] is None
    rows[0]["advantage"] = float("nan")
    with pytest.raises(ValueError):
        save_rollout_audit(tr, rows, tmp_path / "invalid.pt", rollout_provenance={})
    assert not (tmp_path / "invalid.pt").exists()
