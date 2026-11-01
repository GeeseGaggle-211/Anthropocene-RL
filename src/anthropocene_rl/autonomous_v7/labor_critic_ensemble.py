"""Critic ensembles for ``labor_pretrain_eval critic-calibration``.

Several critic checkpoints are scored as ONE averaged critic through the unchanged
single-checkpoint metrics. The averaging is done on per-row predictions, in value units,
before any metric reads them:

* ``qol_years``: the mean of each member's predicted value (QoL-years; for the bounded
  head the decoded ``m * W`` of each member, never a logit or probability average).
* ``population_years``: the mean of each member's expected value (the symexp-decoded
  two-hot expectation times the reference scale).
* ``feasibility`` / ``settlement_guard``: the mean of each member's cost prediction.
* Distribution readers (the ``value_support`` tail terms and
  ``population_tail_explanation``): the uniform mixture of the members' categorical
  distributions. Every tail term is linear in the probabilities, so the mixture's terms
  equal the members' mean term, and the mixture's expectation equals the members' mean
  expected value.

Members are evaluated one after another over the dataset in batches; only per-row
prediction arrays (and, for the population tail explanation, one ``rows x bins`` mixture
array, the same size the single-checkpoint path already holds) are kept, never N copies of
intermediate activations.

An ensemble mixes only checkpoints of one input kind and one set of corrections:
privileged versus plain critics, the observation normalizer, the privileged normalizer,
the value support, the QoL head declaration and the active loader corrections must all
agree, otherwise :func:`load_critic_ensemble` raises ``ValueError`` (fail closed).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ENSEMBLE_SCHEMA = "labor-critic-ensemble-v1"

AVERAGING_RULE: dict[str, Any] = {
    "rule": "uniform mean over members of per-row decoded predictions, formed before any metric",
    "weights": "uniform (1/N)",
    "qol_years": "mean of each member's predicted value in QoL-years "
    "(bounded head: decoded m * W per member; no logit or probability averaging)",
    "population_years": "mean of each member's expected value "
    "(symexp-decoded two-hot expectation x reference scale)",
    "cost_heads": "mean of each member's cost prediction (feasibility, settlement_guard)",
    "distribution_diagnostics": "value_support tail terms and population_tail_explanation read "
    "the uniform mixture of the members' categorical distributions; each tail term is linear "
    "in the probabilities, so it equals the members' mean term",
    "metrics": "all gates, strata, ordering, n-step and TD sections are the single-checkpoint "
    "code applied to the averaged predictions, unchanged",
}


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class EnsembleMember:
    index: int
    path: Path
    sha256: str
    state: Mapping[str, Any]
    task: Any
    costs: Any
    normalizer: Any
    corrections: dict[str, str]
    corrections_source: str


class EnsembleTask:
    """Task-critic stand-in for the shared configuration the report reads.

    Not a module: prediction goes through :func:`ensemble_critic_predictions`. Attributes
    are those of the (validated identical) members' task critics."""

    def __init__(self, members: Sequence[EnsembleMember]) -> None:
        if not members:
            raise ValueError("a critic ensemble needs at least one member")
        self.members = list(members)

    @property
    def _first(self) -> Any:
        return self.members[0].task

    @property
    def support(self) -> Any:
        return self._first.support

    @property
    def qol_head(self) -> Any:
        return self._first.qol_head

    @property
    def bins(self) -> torch.Tensor:
        return self._first.bins

    @property
    def privileged_layout(self) -> Any:
        return self._first.privileged_layout

    @property
    def privileged_normalizer(self) -> Any:
        return self._first.privileged_normalizer

    def is_bounded(self, objective: str) -> bool:
        return self._first.is_bounded(objective)

    def to(self, device: Any) -> EnsembleTask:
        for member in self.members:
            member.task.to(device)
        return self


class EnsembleCosts:
    """Cost-critic counterpart of :class:`EnsembleTask` (same members)."""

    def __init__(self, members: Sequence[EnsembleMember]) -> None:
        self.members = list(members)

    def to(self, device: Any) -> EnsembleCosts:
        for member in self.members:
            member.costs.to(device)
        return self


@dataclass
class CriticEnsemble:
    members: list[EnsembleMember]
    task: EnsembleTask
    costs: EnsembleCosts
    normalizer: Any
    corrections: dict[str, str]
    corrections_source: str
    shared: dict[str, Any]

    def block(self) -> dict[str, Any]:
        """The report's ``ensemble`` record: members, hashes, averaging rule, shared settings."""
        members = []
        for m in self.members:
            fit = m.state.get("heuristic_critic_pretraining") or {}
            members.append(
                {
                    "index": m.index,
                    "checkpoint": str(m.path),
                    "sha256": m.sha256,
                    "fit": {
                        key: fit.get(key) for key in ("dataset_digest", "return_tail", "epochs")
                    },
                }
            )
        return {
            "schema": ENSEMBLE_SCHEMA,
            "member_count": len(self.members),
            "members": members,
            "averaging": dict(AVERAGING_RULE),
            "validated_identical": dict(self.shared),
            "loader_corrections": dict(self.corrections),
            "loader_corrections_source": self.corrections_source,
        }


def _mismatch(what: str, a: EnsembleMember, b: EnsembleMember, left: Any, right: Any) -> ValueError:
    return ValueError(
        f"critic ensemble members must share {what}: member {a.index} ({a.path}) has "
        f"{left!r} but member {b.index} ({b.path}) has {right!r}; ensembles mix only "
        "checkpoints of the same input kind, normalizers, value support and loader corrections"
    )


def load_critic_ensemble(
    paths: Sequence[Path | str],
    corrections_override: Mapping[str, str | None] | None = None,
) -> CriticEnsemble:
    """Load and validate several critic checkpoints as one ensemble (fail closed)."""
    from .labor_pretrain_eval import _load_models, resolve_loader_corrections

    paths = [Path(p) for p in paths]
    if len(paths) < 2:
        raise ValueError("a critic ensemble needs at least two checkpoints")
    members: list[EnsembleMember] = []
    for index, path in enumerate(paths):
        state, _actor, task, costs, normalizer = _load_models(path)
        corrections, source = resolve_loader_corrections(state, corrections_override)
        members.append(
            EnsembleMember(
                index=index,
                path=path,
                sha256=file_sha256(path),
                state=state,
                task=task,
                costs=costs,
                normalizer=normalizer,
                corrections=corrections,
                corrections_source=source,
            )
        )
    return validate_ensemble(members)


def validate_ensemble(members: Sequence[EnsembleMember]) -> CriticEnsemble:
    """Check that members are interchangeable inputs to one averaged critic."""
    from .labor_training import _normalization_digest

    members = list(members)
    first = members[0]
    kind = "privileged" if first.task.privileged_layout is not None else "plain"
    shared: dict[str, Any] = {"input_kind": kind}
    first_digest = _normalization_digest(first.normalizer)
    for other in members[1:]:
        other_kind = "privileged" if other.task.privileged_layout is not None else "plain"
        if other_kind != kind:
            raise _mismatch(
                "input kind (privileged vs plain critic)", first, other, kind, other_kind
            )
        digest = _normalization_digest(other.normalizer)
        if digest != first_digest:
            raise _mismatch("the observation normalizer", first, other, first_digest, digest)
        if other.task.support != first.task.support:
            raise _mismatch(
                "the value support", first, other, first.task.support, other.task.support
            )
        if other.task.qol_head != first.task.qol_head:
            raise _mismatch(
                "the QoL head declaration", first, other, first.task.qol_head, other.task.qol_head
            )
        if not torch.allclose(
            first.task.bins.detach().cpu().double(), other.task.bins.detach().cpu().double()
        ):
            raise _mismatch("the support bins", first, other, "bins", "different bins")
        if other.corrections != first.corrections:
            raise _mismatch(
                "the active loader corrections (tail normalization, false-terminal policy; "
                "pass --tail-normalize/--false-terminal-policy to apply one explicitly to all)",
                first,
                other,
                first.corrections,
                other.corrections,
            )
        if other.corrections_source != first.corrections_source:
            raise _mismatch(
                "the loader-correction source",
                first,
                other,
                first.corrections_source,
                other.corrections_source,
            )
    for member in members:
        if (member.costs.privileged_layout is not None) != (
            member.task.privileged_layout is not None
        ):
            raise ValueError(
                f"member {member.index} ({member.path}): task and cost critics differ in input kind"
            )
    if kind == "privileged":
        digests = []
        for member in members:
            if member.task.privileged_layout != first.task.privileged_layout:
                raise _mismatch(
                    "the privileged layout",
                    first,
                    member,
                    first.task.privileged_layout,
                    member.task.privileged_layout,
                )
            digests.append(member.task.privileged_normalizer.digest())
            if member.costs.privileged_normalizer.digest() != digests[-1]:
                raise ValueError(
                    f"member {member.index} ({member.path}): task and cost critics carry "
                    "different privileged normalizers"
                )
        for member, digest in zip(members, digests, strict=True):
            if digest != digests[0]:
                raise _mismatch("the privileged normalizer", first, member, digests[0], digest)
        shared["privileged_normalizer_digest"] = digests[0]
    shared["observation_normalizer_digest"] = first_digest
    shared["value_support"] = first.task.support.state_dict()
    shared["qol_head"] = None if first.task.qol_head is None else first.task.qol_head.state_dict()
    return CriticEnsemble(
        members=members,
        task=EnsembleTask(members),
        costs=EnsembleCosts(members),
        normalizer=first.normalizer,
        corrections=dict(first.corrections),
        corrections_source=first.corrections_source,
        shared=shared,
    )


def ensemble_critic_predictions(
    task: EnsembleTask,
    costs: EnsembleCosts,
    dataset: Mapping[str, Any],
    rows: np.ndarray,
    batch_size: int = 256,
    observed_max: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Per-row mean over members of :func:`labor_pretrain_eval.critic_predictions`.

    Members run in sequence; only the running float64 sums are kept. The tail-diagnostic
    keys (``"<head>/<term>"``) average too, which is the mixture's term (linear)."""
    from .labor_pretrain_eval import critic_predictions

    if not isinstance(costs, EnsembleCosts) or [m.index for m in costs.members] != [
        m.index for m in task.members
    ]:
        raise ValueError("ensemble task and cost critics must be the same members")
    total: dict[str, np.ndarray] | None = None
    for member in task.members:
        part = critic_predictions(
            member.task, member.costs, dataset, rows, batch_size, observed_max
        )
        if total is None:
            total = {k: np.array(v, dtype=np.float64) for k, v in part.items()}
        else:
            if part.keys() != total.keys():
                raise ValueError("ensemble members produced different prediction heads")
            for key, value in part.items():
                total[key] += value
        del part
    assert total is not None
    count = float(len(task.members))
    return {key: value / count for key, value in total.items()}


def ensemble_head_probabilities(
    task: EnsembleTask,
    dataset: Mapping[str, Any],
    rows: np.ndarray,
    head: str,
    batch_size: int,
) -> np.ndarray:
    """Uniform mixture of the members' categorical distributions for ``head`` (rows x bins)."""
    from .labor_pretrain_eval import _head_logits

    total: np.ndarray | None = None
    for member in task.members:
        logits = _head_logits(member.task, dataset, rows, head, batch_size)
        probabilities = torch.softmax(logits, dim=-1).numpy()
        total = (
            probabilities.astype(np.float64, copy=True) if total is None else total + probabilities
        )
    assert total is not None
    return total / float(len(task.members))
