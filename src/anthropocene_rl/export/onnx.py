"""Candidate ONNX export backend for a promoted actor.

ONNX is not yet the selected Anthropocene runtime actor format. Keep export
backend-specific code isolated so a later runtime decision does not redefine the
training or native environment contracts.
"""


def export_actor() -> None:
    raise NotImplementedError("Export follows successful training and runtime parity review")
