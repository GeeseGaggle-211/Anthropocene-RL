"""Gymnasium facade over the native C# Elder environment.

Do not implement simulation logic here. Reset/Step must delegate to the native
process boundary and preserve native termination, truncation, masks, reward,
interventions, and replay metadata.
"""

from __future__ import annotations


class AnthropoceneElderEnv:
    """Placeholder for the section-34 Gymnasium adapter."""

    def __init__(self) -> None:
        raise NotImplementedError("Transport and action contract are not locked")
