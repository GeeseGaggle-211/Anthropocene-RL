"""Prefix comparisons must use the native collector's accumulation semantics."""

import importlib.util
import sys
from pathlib import Path


def test_prefix_welfare_preserves_native_sequential_float_accumulation():
    directory = Path("scripts").resolve()
    spec = importlib.util.spec_from_file_location(
        "horizon_welfare_fixed", directory / "evaluate_autonomous_horizon_v5_fixed.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(directory))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(directory))
    # Exact physical interval rewards from preserved seed29 entry development episode.
    rewards = [
        1858.5203748738943,
        1812.7994037490062,
        1863.1045627231686,
        1746.3263142949504,
        1127.6218721102841,
        571.1686567811239,
        790.1507383012324,
        625.6315600494387,
    ]
    frames = [dict(raw_welfare=value) for value in rewards]
    assert module.sequential_welfare(frames) / 100 == 103.95323482883097
    if sys.version_info >= (3, 12):
        assert sum(rewards) / 100 != 103.95323482883097
