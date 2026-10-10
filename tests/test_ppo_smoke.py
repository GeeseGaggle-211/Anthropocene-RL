import math

import pytest

pytest.importorskip("torch")

from anthropocene_rl.training.smoke import run_ppo_smoke  # noqa: E402


def test_ppo_mechanics_smoke_is_finite() -> None:
    metrics = run_ppo_smoke(batch_size=32)
    assert metrics
    assert all(math.isfinite(value) for value in metrics.values())
