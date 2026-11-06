"""Paired-branch runner for the A(s,a) work (stage 2 warm-up and the horizon check).

From one forked root state, run two first actions ``a`` and ``b`` for ``k`` decisions
under the SAME common random numbers and return, per branch and stream, records in
exactly the export-v2 ``successor_streams`` format (``labor_search_payload.successor_stream``)
plus the focal camp's critic-input row at EVERY decision (``labor_sa_advantage.
parse_successors`` reads :meth:`PairedBranches.raw_root` unchanged).

Everything rests on ``NativeSearch`` (labor_search): fork/restore of the simulator,
``extend`` (the branch loop), the keyed continuation dither and ``record_boundary``.
This module adds no simulator access of its own.

Common random numbers
---------------------
Both branches restore the SAME ``root_fork`` and use the SAME ``root_key``. Stream 0 is
undithered; stream ``s >= 1`` dithers the other camps (and the focal camp's H12
continuation) with ``default_rng([DITHER_TAG, *root_key, s, decision, camp])``, which
does not depend on the branch or on the action. Hence two identical actions produce
bitwise-identical branches, and any difference between branches is caused by the action.

Continuation policy interface
-----------------------------
``continuation`` is ``None`` (H12 continues, as in the search) or a callable

    continuation(decision: int, label, catalog, step) -> payload | None

called at decisions >= 1 for the focal camp only. ``label`` is the converted H12 label
for the focal camp at that state (``label.action`` is the H12 payload), ``catalog`` its
``LaborCatalog``, ``step`` the simulator step (read-only). Return the labor payload to
execute (it is exempt from the dither, like a first action) or ``None`` to defer to
H12. It must be a pure function of its arguments and of keyed randomness: it must not
carry hidden state between calls, because it is invoked once per branch and stream and
the branches must see identical draws. ``KeyedActorContinuation`` wraps an
``act(decision, label, catalog, step, rng)`` callable and supplies
``rng = default_rng([ACTOR_TAG, *key, decision])`` (the same for every branch).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import contracts
from .labor_search import COMPONENTS, Branch, NativeSearch
from .labor_search_payload import SuccessorRows, successor_stream

ACTOR_TAG = 0x4142  # "AB": paired-branch actor continuation draws
DEFAULT_K = 3

ContinuationPolicy = Callable[[int, Any, Any, Mapping[str, Any]], "Mapping[str, Any] | None"]


def h12_continuation(decision: int, label: Any, catalog: Any, step: Mapping[str, Any]) -> None:
    """The search's continuation: always defer to H12 (``None``)."""
    return None


class KeyedActorContinuation:
    """Adapt ``act(decision, label, catalog, step, rng) -> payload | None`` to the
    continuation interface with a branch-independent keyed ``rng`` (CRN-safe)."""

    def __init__(self, act: Callable[..., Mapping[str, Any] | None], key: Sequence[int]) -> None:
        self.act = act
        self.key = [int(x) for x in key]

    def __call__(self, decision: int, label: Any, catalog: Any, step: Mapping[str, Any]):
        rng = np.random.default_rng([ACTOR_TAG, *self.key, int(decision)])
        return self.act(decision, label, catalog, step, rng)


@dataclass
class PairedBranches:
    """Both branches of one pair. ``streams[name]`` lists ``successor_stream`` dicts
    (root-local row indices into ``store.rows``, shared by both branches)."""

    k: int
    stream_ids: tuple[int, ...]
    streams: dict[str, list[dict[str, Any]]]
    store: SuccessorRows
    resolves: int = 0
    decisions: int = 0
    discount_years: float = contracts.HORIZON_YEARS
    identical_actions: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def raw_root(self) -> dict[str, Any]:
        """A raw root in the export-v2 layout (candidate 0 = ``a``, 1 = ``b``) that
        ``labor_sa_advantage.parse_successors`` reads unchanged; rows live in ``store``."""
        return {
            "candidates": [
                {"index": 0, "successor_streams": self.streams["a"]},
                {"index": 1, "successor_streams": self.streams["b"]},
            ]
        }

    def stream(self, name: str, stream: int) -> dict[str, Any]:
        return next(s for s in self.streams[name] if int(s["stream"]) == int(stream))


def run_paired_branches(
    native: NativeSearch,
    *,
    root_fork: int,
    root_h12: Mapping[int, Any],
    root_key: Sequence[int],
    focal: int,
    society: int,
    founded: Mapping[int, int],
    first_a: Mapping[str, Any],
    first_b: Mapping[str, Any],
    personality: Sequence[float] = (0.5, 0.5),
    k: int = DEFAULT_K,
    continuation: ContinuationPolicy | None = None,
    streams: Sequence[int] = (0,),
) -> PairedBranches:
    """Run first actions ``a`` and ``b`` for ``k`` decisions on each of ``streams`` from
    the forked root, under common random numbers. ``native.successor_view`` must be set
    (the focal critic-input row at each successor). ``native`` is left as found:
    the root fork is neither dropped nor modified, and the row-every-decision switch is
    restored. A terminated focal camp ends its branch early (``terminated``)."""
    if int(k) < 1:
        raise ValueError("k must be at least 1")
    stream_ids = tuple(int(s) for s in streams)
    if not stream_ids or len(set(stream_ids)) != len(stream_ids) or min(stream_ids) < 0:
        raise ValueError("streams must be distinct non-negative integers")
    if native.successor_view is None:
        raise ValueError("native.successor_view must be set to record successor rows")
    store = SuccessorRows(personality)
    out: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}
    before = native.work.resolves
    labor_before = native.work.labor_steps
    previous = native.successor_every_decision
    native.successor_every_decision = True
    try:
        for name, candidate in (("a", first_a), ("b", first_b)):
            for stream in stream_ids:
                branch = Branch(
                    0 if name == "a" else 1,
                    stream,
                    founded=dict(founded),
                    societies={int(society)},
                    record=True,
                )
                native.extend(
                    branch,
                    int(k),
                    root_fork=root_fork,
                    focal=focal,
                    candidate=candidate,
                    root_h12=root_h12,
                    root_key=list(root_key),
                    keep_fork=False,
                    focal_policy=continuation,
                )
                out[name].append(
                    successor_stream(
                        stream,
                        branch.steps,
                        branch.years,
                        branch.terminated,
                        branch.boundaries,
                        store,
                    )
                )
    finally:
        native.successor_every_decision = previous
    return PairedBranches(
        k=int(k),
        stream_ids=stream_ids,
        streams=out,
        store=store,
        resolves=native.work.resolves - before,
        decisions=native.work.labor_steps - labor_before,
        discount_years=float(native.config.discount_years),
        identical_actions=dict(first_a) == dict(first_b),
    )


def cumulative_returns(
    stream: Mapping[str, Any], k: int, discount_years: float = contracts.HORIZON_YEARS
) -> np.ndarray:
    """Discounted cumulative return after decision ``j`` for j = 1..k, shape [k, 4]
    (same discounting as ``Branch.returns``). A branch that terminated earlier is
    closed with zero future: its cumulative value is held constant after termination."""
    rewards = np.asarray(stream["step_rewards"], dtype=np.float64).reshape(-1, len(COMPONENTS))
    years = np.asarray(stream["step_years"], dtype=np.float64)
    out = np.zeros((int(k), len(COMPONENTS)))
    total = np.zeros(len(COMPONENTS))
    discount = 1.0
    for j in range(int(k)):
        if j < len(rewards):
            total = total + discount * rewards[j]
            discount *= math.exp(-float(years[j]) / discount_years)
        elif not stream["terminated"]:
            raise ValueError("a non-terminated branch is shorter than k")
        out[j] = total
    return out


def paired_differences(pair: PairedBranches, k: int | None = None) -> np.ndarray:
    """Cumulative paired difference (a - b) at every k' = 1..k, [S, k, 4] over the
    pair's streams (S = len(pair.stream_ids), in that order)."""
    k = pair.k if k is None else int(k)
    if k > pair.k:
        raise ValueError("k exceeds the horizon the pair was run to")
    diffs = []
    for stream in pair.stream_ids:
        a = cumulative_returns(pair.stream("a", stream), k, pair.discount_years)
        b = cumulative_returns(pair.stream("b", stream), k, pair.discount_years)
        diffs.append(a - b)
    return np.stack(diffs)
