"""fit-normalizer refuses non-fitting worlds and takes r5 reset options (r5 manifest section 10)."""

from __future__ import annotations

import pytest

from anthropocene_rl.autonomous_v7 import labor_training


def _args(worlds: str):
    return labor_training.argument_parser().parse_args(
        ["fit-normalizer", "--worlds", worlds, "--turns", "1", "--out", "unused.pt"]
    )


@pytest.mark.parametrize("worlds", ["90154", "90180", "2001", "3000"])
def test_fit_normalizer_refuses_non_fitting_worlds(worlds: str) -> None:
    with pytest.raises(Exception):  # noqa: B017 - blind Exception assertion kept (test expectation unchanged)
        labor_training._command_fit_normalizer(_args(worlds))


def test_fit_normalizer_accepts_reset_options() -> None:
    args = labor_training.argument_parser().parse_args(
        [
            "fit-normalizer",
            "--worlds",
            "90100",
            "--players",
            "16",
            "--width",
            "128",
            "--height",
            "64",
            "--separation",
            "9",
            "--out",
            "unused.pt",
        ]
    )
    options = labor_training._labor_reset_options(args)
    assert (
        options.players == 16
        and options.movement
        and options.width == 128
        and options.separation == 9
    )
