"""Episode separation and exact repeated-reset inputs for stochastic diagnostics."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_autonomous_critic_coverage_v12 import repeated_openings, returns_for


def test_replicated_reset_returns_are_separate_and_observations_exact():
    rows = [
        dict(reward=r, elapsed=20, terminal=True, value=3.0, obs={"x": torch.ones(1, 2)})
        for r in [2.0, 8.0]
    ]
    eps = [dict(id="same-reset", steps=1, tail_value=999.0) for _ in rows]
    ys = returns_for(rows, eps)
    assert ys["target"] == [2.0, 8.0]
    assert ys["zero_tail"] == ys["doubled_tail"] == ys["target"]
    result = repeated_openings(rows, eps, ys)["same-reset"]
    assert result["target_mean"] == 5.0
    assert result["target_sample_variance"] == 18.0
    rows[1]["obs"]["x"] = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="reset inputs"):
        repeated_openings(rows, eps, ys)
