"""Final readiness cannot be inferred from a still-growing campaign summary."""

import copy
import runpy
from pathlib import Path

import pytest

validate_complete = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "scripts/run_autonomous_continuation_diagnostics_v4.py"
    )
)["validate_complete"]


def test_diagnostics_require_both_terminal_seeds_and_consistent_counts():
    campaign = dict(
        config=dict(seeds=[29, 43], additional_transitions_per_seed=24576),
        seeds=[
            dict(seed=seed, status="completed", additional_transitions=24576) for seed in [29, 43]
        ],
        additional_transitions=49152,
    )
    validate_complete(campaign)
    invalid = copy.deepcopy(campaign)
    invalid["seeds"] = invalid["seeds"][:1]
    invalid["additional_transitions"] = 24576
    with pytest.raises(ValueError):
        validate_complete(invalid)
    invalid = copy.deepcopy(campaign)
    invalid["seeds"][1]["seed"] = 29
    with pytest.raises(ValueError):
        validate_complete(invalid)
    invalid = copy.deepcopy(campaign)
    invalid["seeds"][1]["additional_transitions"] = 4096
    invalid["additional_transitions"] = 28672
    with pytest.raises(ValueError):
        validate_complete(invalid)
    invalid["seeds"][1]["status"] = "recovery_probe_pause"
    validate_complete(invalid)  # A pause needs diagnosis; it is not an automatic readiness pass.
    invalid["additional_transitions"] -= 1
    with pytest.raises(ValueError):
        validate_complete(invalid)
