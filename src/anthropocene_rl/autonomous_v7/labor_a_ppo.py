"""WP-A1: the stage-1 A(s,a) head as a Q-Prop-style per-head control variate in labor PPO.

Amendment B (adopted 8 Nov 2026). Nothing here trains A or the critic it reads.

B1 -- a frozen critic snapshot. :func:`load_frozen_a` loads ``--a-critic`` (a labor
checkpoint; in production the stage-1 critic privileged-e4 seed 0) as an eval-mode,
no-grad context provider and ``--a-warm-start`` (the ``labor_a_stage1`` warm-start
artifact) against it with ``labor_advantage.load_advantage_head`` (its
``labor_critic_digest`` check is never relaxed). The online V that GAE uses is a different
object. :meth:`FrozenA.assert_unchanged` re-checks both digests after an iteration.

Where A is computed. In ``train`` (not at collect), from the batch's stored observations
(``initial_tensors``/``initial_privileged_raw``), retained samples and the parent's
pre-action population (rollout ``decisionRows[i]["population"]``), once per iteration
before the actor update, by :func:`prepare_iteration`, with the pre-update actor (which
is the behaviour policy: ``train`` checks ``behavior_policy_digest``).

Units and combination. A predicts per value component (``COMPONENTS``) in head units.
Every A quantity is converted with ``labor_a_stage2.to_raw_units`` and scalarised with
``labor_ppo.combine_labor_advantages`` using the batch's persisted ``cost_multipliers``
and per-row personality weights (``batch.row_personality_weights``): exactly how the GAE
advantage it is subtracted from was combined. ``update_labor_policy`` does not normalise
advantages, so the control variate needs no normalisation step and eta = 0 reproduces
the GAE coefficients exactly. A's per-root centring is irrelevant: every use is a
difference A(x) - A(y) at one root or a probability-weighted sum whose weights sum to 1.

B2 -- per policy head h (other heads held at their sampled values; the actor's heads are
conditionally independent given the state, so this is the factorised case):

* movement, fission: E_h = sum_k pi_h(k) A(k), exact over the head's categories
  (``labor_a_stage2.categorical_expectation``, gradient through pi_h only).
* trade_gate is categorical PER TRADE GROUP (7 groups x 3 gates, 3**7 joint categories).
  It is enumerated exactly per group with the other groups at their sampled values (the
  local-expectation estimator: unbiased for grad E_{trade_gate}[A]); its score-function
  term is split into the seven group factors, each with its own coefficient
  ``GAE - eta (A(s,a) - E_g)`` (``labor_ppo.InitialCorrection.sub_heads``).
* specialty, activity, trade_value, trade_share: M (default 16) draws x_m from head h's
  distribution at the start of the iteration (keyed stream ``DRAW_TAG``), each encoded
  with the NumPy action encoder (apportionment included) and scored by A. The correction
  is ``mean_m w_m grad log pi_h(x_m) (A(x_m) - b_m)``, with ``b_m`` the leave-one-out mean
  of the other M - 1 scores and ``w_m = pi_h(x_m) / pi_h^start(x_m)`` (= 1 at the
  iteration start; it keeps the estimator unbiased for the current policy in later
  epochs). The coefficient baseline is E_h = mean_m A(x_m) (independent of the executed
  action, so unbiased). A head whose action blocks are all masked in A
  (``LaborAdvantageHead.masked_blocks``; trade_value/trade_share in stage 1) is "A-blind":
  A(x) does not depend on it, so its residual and correction are exactly 0 and no draws
  are made.

Head h's surrogate coefficient is ``GAE - eta_h (A(s,a) - E_h)`` with E_h fixed at the
iteration start, plus ``eta_h E_h`` (current policy) added to the initial-edge objective.
Retry edges and non-initial terms use eta = 0 (unchanged).

B3/D7 -- admission (lagged). ``train`` measures, on ``d7_rows`` rows of iteration i's batch
(``fixed_audit_indices``) with the pre-update actor, each head's per-row gradient-estimate
variance (trace of the covariance over actor parameters) for the V-only estimate
``grad log pi_h * GAE`` and the control-variate estimate; eta_h = 1 at iteration i + 1 iff
``var_v > 0`` and ``1 - var_cv / var_v >= D7_MIN_VARIANCE_REDUCTION``. The variance is the
exact trace (streamed per row, no random projection): the same quantity as
``labor_a_stage2.gradient_variance`` on unprojected per-row gradients, in O(params) memory.

B5 -- :func:`warmup_report` builds the ``labor_a_stage2.WarmupReport`` telemetry; it never
changes training.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from . import contracts
from . import labor_a_stage2 as s2
from . import labor_advantage as la
from .labor_model import (
    LABOR_COSTS,
    LABOR_OBJECTIVES,
    LaborPersonality,
    labor_log_prob,
    nonlabor_log_prob,
    sample_labor_policy,
    sample_nonlabor_policy,
    schema12_initial_action,
)
from .labor_ppo import (
    INITIAL_POLICY_HEADS,
    LABOR_POLICY_HEADS,
    NONLABOR_POLICY_HEADS,
    InitialCorrection,
    InitialCorrectionFn,
    LaborAttemptPolicyBatch,
    _rows,
    combine_labor_advantages,
)
from .ppo_correctness import fixed_audit_indices

A_MODES = ("off", "gae")
A_STATE_KEY = "a_control_variate"
A_STATE_SCHEMA = "labor-a-control-variate-state-v1"
# D7: admit A as head h's control variate when it cuts the head's policy-gradient
# variance by at least this fraction on held-out rows. Provisional (Amendment B).
D7_MIN_VARIANCE_REDUCTION = 0.25
DEFAULT_DRAWS = 16  # M, draws per row for the sampled heads
DEFAULT_D7_ROWS = 64  # rows of the measured batch used for the D7 variance
DRAW_TAG = 0x4144  # "AD": keyed control-variate draws for the sampled heads

CATEGORICAL_HEADS = ("movement", "fission", "trade_gate")
SAMPLED_HEADS = ("specialty", "activity", "trade_value", "trade_share")
FACTORED_HEADS = ("trade_gate",)  # categorical per trade group; enumerated per group
assert set(CATEGORICAL_HEADS) | set(SAMPLED_HEADS) == set(INITIAL_POLICY_HEADS)
assert not set(CATEGORICAL_HEADS) & set(SAMPLED_HEADS)
# Action blocks each policy head can change in the encoded action. Specialty also
# changes the activity block (unstaffed roles are encoded as idle after apportionment).
HEAD_BLOCKS: dict[str, tuple[str, ...]] = {
    "specialty": ("specialty_share", "activity_share"),
    "activity": ("activity_share",),
    "movement": ("movement_onehot",),
    "fission": ("fission_onehot",),
    "trade_gate": ("trade_gate_onehot",),
    "trade_value": ("trade_value_log",),
    "trade_share": ("trade_share",),
}
_LABOR_MASK_KEYS = (
    "labor_specialty_mask",
    "labor_activity_mask",
    "labor_specialty_present",
    "labor_activity_present",
)
_SCORE_CHUNK = 4096
_FORWARD_CHUNK = 512


def zero_eta() -> dict[str, float]:
    return {h: 0.0 for h in INITIAL_POLICY_HEADS}


# ---------------------------------------------------------------------------
# B1: the frozen A and its critic
# ---------------------------------------------------------------------------


def _module_digest(module: torch.nn.Module) -> str:
    from .economic_trace import state_digest

    return state_digest(module)


@dataclass
class FrozenA:
    """A frozen critic snapshot (context provider) and the A head fitted on it."""

    task_critic: torch.nn.Module
    cost_critics: torch.nn.Module
    head: la.LaborAdvantageHead
    critic_digest: str
    head_digest: str
    warm_start: Mapping[str, Any]

    def blind_heads(self) -> tuple[str, ...]:
        """Policy heads whose every action block is masked in A (A ignores them)."""
        masked = set(self.head.masked_blocks)
        return tuple(h for h in INITIAL_POLICY_HEADS if set(HEAD_BLOCKS[h]) <= masked)

    def contexts(
        self,
        tensors: Mapping[str, torch.Tensor],
        privileged_raw: Mapping[str, Any] | None,
    ) -> dict[str, torch.Tensor]:
        if self.task_critic.privileged_layout is not None and privileged_raw is None:
            raise ValueError("the frozen A critic is privileged; the rollout has no raw arrays")
        return la.critic_contexts(self.task_critic, self.cost_critics, tensors, privileged_raw)

    def score(
        self,
        contexts: Mapping[str, torch.Tensor],
        features: torch.Tensor,
        root_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Raw A per component (head units) for candidate ``features`` [K, 349]."""
        out: dict[str, list[torch.Tensor]] = defaultdict(list)
        with torch.no_grad():
            for start in range(0, len(features), _SCORE_CHUNK):
                stop = min(len(features), start + _SCORE_CHUNK)
                part = self.head(contexts, features[start:stop], root_index[start:stop])
                for name, value in part.items():
                    out[name].append(value.detach())
        if not len(features):
            return {name: torch.zeros(0) for name in la.COMPONENTS}
        return {name: torch.cat(out[name]) for name in la.COMPONENTS}

    def assert_unchanged(self) -> None:
        from .labor_value_calibration import labor_critic_digest

        if labor_critic_digest(self.task_critic, self.cost_critics) != self.critic_digest:
            raise AssertionError("the frozen A critic changed during the iteration")
        if _module_digest(self.head) != self.head_digest:
            raise AssertionError("the frozen A head changed during the iteration")
        if any(p.requires_grad for p in self._parameters()):
            raise AssertionError("a frozen A parameter requires grad")

    def _parameters(self) -> list[torch.nn.Parameter]:
        return [
            *self.task_critic.parameters(),
            *self.cost_critics.parameters(),
            *self.head.parameters(),
        ]

    def record(self) -> dict[str, Any]:
        return {
            "criticDigest": self.critic_digest,
            "headDigest": self.head_digest,
            "maskedBlocks": list(self.head.masked_blocks),
            "aBlindHeads": list(self.blind_heads()),
            "warmStartDecisions": dict(self.warm_start.get("decisions", {})),
        }


def load_frozen_a(
    warm_start: Mapping[str, Any],
    critic_state: Mapping[str, Any],
    *,
    normalization_binding_digest: str,
) -> FrozenA:
    """Freeze the critic in ``critic_state`` and load the warm-start A against it.

    ``normalization_binding_digest`` is the rollout's observation-normalizer binding: A's
    contexts read the batch's stored (already normalised) tensors, so the frozen critic
    must have been trained under the same observation normalizer. Refuses otherwise.
    """
    from .labor_a_stage1 import WARM_START_SCHEMA
    from .labor_training import _models_from_checkpoint, _normalization_digest
    from .labor_value_calibration import labor_critic_digest

    if not isinstance(warm_start, Mapping):
        raise ValueError("--a-warm-start is not a labor advantage-head artifact")
    block = warm_start.get("warm_start")
    if not isinstance(block, Mapping) or block.get("schema") != WARM_START_SCHEMA:
        raise ValueError("--a-warm-start must be a labor_a_stage1 warm-start artifact")
    _, _, task, costs, normalizer = _models_from_checkpoint(critic_state)
    if _normalization_digest(normalizer) != normalization_binding_digest:
        raise ValueError(
            "the frozen A critic was trained under another observation normalizer than the "
            "rollout's; its contexts would read mis-normalised inputs"
        )
    for module in (task, costs):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    # Refuses a critic whose labor_critic_digest differs from the fitted one.
    head, _ = la.load_advantage_head(warm_start, task, costs)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    decisions = block.get("decisions", {})
    for name in la.COMPONENTS:
        if decisions.get(name) == "zero":
            layer = head.heads[name][2]
            if bool(layer.weight.abs().sum() != 0) or bool(layer.bias.abs().sum() != 0):
                raise ValueError(f"warm-start head {name} is declared zero but is not")
    return FrozenA(
        task,
        costs,
        head,
        labor_critic_digest(task, costs),
        _module_digest(head),
        {"decisions": dict(decisions), "status": block.get("status")},
    )


# ---------------------------------------------------------------------------
# Units: head-unit components -> the GAE's combined policy unit
# ---------------------------------------------------------------------------


def combine_head_units(
    values: Mapping[str, torch.Tensor],
    row_weights: torch.Tensor,
    cost_multipliers: Mapping[str, float],
) -> torch.Tensor:
    """A per component (head units, any shape [N, ...]) -> combined policy unit [N, ...].

    ``to_raw_units`` then ``combine_labor_advantages`` with the GAE's per-row personality
    weights and cost multipliers (row i's weights apply to every candidate of row i).
    """
    first = values[la.COMPONENTS[0]]
    shape = first.shape
    if not shape or shape[0] != len(row_weights):
        raise ValueError("A values must be row-aligned with the personality weights")
    per_row = int(np.prod(shape[1:])) if len(shape) > 1 else 1
    raw = s2.to_raw_units({name: values[name].reshape(-1) for name in la.COMPONENTS})
    weights = row_weights.repeat_interleave(per_row, dim=0) if per_row != 1 else row_weights
    combined = combine_labor_advantages(
        {name: raw[name] for name in LABOR_OBJECTIVES},
        {name: raw[name] for name in LABOR_COSTS},
        personality=LaborPersonality(),
        cost_multipliers=cost_multipliers,
        row_weights=weights,
    )
    return combined.reshape(shape)


# ---------------------------------------------------------------------------
# Keyed draws and the sampled-head correction
# ---------------------------------------------------------------------------


def draw_generator(key: Sequence[int], row: int, head: str) -> torch.Generator:
    """Keyed generator for (iteration key, parent row, head): independent of row order."""
    seed = int(
        np.random.default_rng(
            [DRAW_TAG, *(int(k) for k in key), int(row), INITIAL_POLICY_HEADS.index(head)]
        ).integers(2**62)
    )
    return torch.Generator().manual_seed(seed)


def leave_one_out_centred(scores: torch.Tensor) -> torch.Tensor:
    """``A(x_m) - b_m`` with ``b_m`` the mean of the other M - 1 scores, [N, M]."""
    m = scores.shape[-1]
    if m < 2:
        raise ValueError("the leave-one-out baseline needs at least two draws")
    total = scores.sum(dim=-1, keepdim=True)
    return scores - (total - scores) / (m - 1)


def sampled_correction(
    current_logp: torch.Tensor,
    start_logp: torch.Tensor,
    scores: torch.Tensor,
) -> torch.Tensor:
    """Per-row term [N] whose gradient is ``mean_m w_m grad log pi(x_m) (A(x_m) - b_m)``.

    ``current_logp`` [N, M] (differentiable) and ``start_logp`` [N, M] are log pi_h of the
    draws under the current and the drawing policy, ``scores`` [N, M] = A(x_m) (detached).
    ``w_m = exp(current - start)`` is 1 at the drawing policy. The returned VALUE is the
    plain estimate ``mean_m A(x_m)`` of E_h (the gradient carrier has value 0).
    """
    centred = leave_one_out_centred(scores.detach())
    weight = torch.exp(current_logp - start_logp.detach())
    carrier = (weight * centred).mean(dim=-1)
    return scores.detach().mean(dim=-1) + (carrier - carrier.detach())


# ---------------------------------------------------------------------------
# Per-iteration preparation
# ---------------------------------------------------------------------------


@dataclass
class SampledDraws:
    head: str
    draws: int
    sample: dict[str, torch.Tensor]  # spliced labor or non-labor sample, rows [N * M]
    start_logp: torch.Tensor  # [N, M]
    scores: torch.Tensor  # [N, M] combined A of each draw


@dataclass
class AIteration:
    """A's quantities for one batch, computed once before the actor update."""

    eta: dict[str, float]
    advantage: torch.Tensor  # [N] the stored GAE (combined unit)
    a_executed: torch.Tensor  # [N] combined A(s, a)
    categorical: dict[str, torch.Tensor]  # movement [N,17], fission [N,3], trade_gate [N,G,3]
    expected: dict[str, torch.Tensor]  # E_h at the start, [N] ([N, G] for trade_gate)
    draws: dict[str, SampledDraws]
    blind: tuple[str, ...]
    batch: LaborAttemptPolicyBatch
    counts: dict[str, int] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)

    # -- coefficients ------------------------------------------------------------

    def coefficients(self, eta: Mapping[str, float] | None = None) -> dict[str, torch.Tensor]:
        """Per-head score coefficient ``GAE - eta_h (A(s,a) - E_h)``, detached [N].

        A factored head (trade_gate) with eta 1 keeps the plain GAE here; its per-group
        coefficients come from :meth:`group_coefficients` via ``InitialCorrection``.
        """
        eta = self.eta if eta is None else eta
        out = {}
        for h in INITIAL_POLICY_HEADS:
            e = float(eta[h])
            if e not in (0.0, 1.0):
                raise ValueError("eta must be 0 or 1 per head")
            if e == 0.0 or h in FACTORED_HEADS or h in self.blind:
                out[h] = self.advantage
            else:
                out[h] = (self.advantage - (self.a_executed - self.expected[h])).detach()
        return out

    def group_coefficients(self, head: str = "trade_gate") -> torch.Tensor:
        """Per-group coefficient ``GAE - (A(s,a) - E_g)`` [N, G] of a factored head."""
        return (
            self.advantage[:, None] - (self.a_executed[:, None] - self.expected[head])
        ).detach()

    # -- differentiable terms ------------------------------------------------------

    def head_terms(
        self,
        output: Mapping[str, torch.Tensor],
        tensors: Mapping[str, torch.Tensor],
        parent_index: torch.Tensor,
        heads: Sequence[str],
    ) -> dict[str, tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]]:
        """Per head: (E_h term [n] with the correction gradient, factored split or None).

        ``output`` is the current actor's output on ``tensors`` (the rows
        ``parent_index`` of the batch). A-blind heads return an exact zero term.
        """
        from .labor_model import _nonlabor_output

        result: dict[str, tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]] = {}
        movement = fission = gate = None
        if any(h in CATEGORICAL_HEADS for h in heads):
            movement, movement_mask, _, fission, _, _, gate, _ = _nonlabor_output(output)
        for h in heads:
            if h in self.blind:
                raise ValueError(f"{h} is A-blind: its control-variate term is exactly zero")
            if h == "movement":
                probs = torch.softmax(movement.masked_fill(~movement_mask, -torch.inf), dim=1)
                values = self.categorical["movement"].index_select(0, parent_index)
                result[h] = (s2.categorical_expectation(probs, values), None)
            elif h == "fission":
                probs = torch.softmax(fission, dim=1)
                values = self.categorical["fission"].index_select(0, parent_index)
                result[h] = (s2.categorical_expectation(probs, values), None)
            elif h == "trade_gate":
                groups = gate.shape[1]
                probs = torch.softmax(gate, dim=2)
                values = self.categorical["trade_gate"].index_select(0, parent_index)
                expected = torch.stack(
                    [
                        s2.categorical_expectation(probs[:, g], values[:, g])
                        for g in range(groups)
                    ],
                    dim=1,
                )
                executed = self.batch.initial_nonlabor_sample["trade_gate"].index_select(
                    0, parent_index
                )
                logp = torch.log_softmax(gate, dim=2).gather(2, executed[..., None]).squeeze(-1)
                coefficient = self.group_coefficients().index_select(0, parent_index)
                result[h] = (expected.sum(dim=1), (logp, coefficient))
            else:
                result[h] = (self._sampled_term(h, output, tensors, parent_index), None)
        return result

    def _sampled_term(
        self,
        head: str,
        output: Mapping[str, torch.Tensor],
        tensors: Mapping[str, torch.Tensor],
        parent_index: torch.Tensor,
    ) -> torch.Tensor:
        draws = self.draws[head]
        m = draws.draws
        n = len(parent_index)
        local = torch.arange(n).repeat_interleave(m)
        rows = (parent_index[:, None] * m + torch.arange(m)[None, :]).reshape(-1)
        sample = _rows(draws.sample, rows)
        output_rep = _rows(output, local)
        if head in LABOR_POLICY_HEADS:
            masks = _rows({k: tensors[k] for k in _LABOR_MASK_KEYS}, local)
            logp = labor_log_prob(output_rep, sample, masks)[head]
        else:
            logp = nonlabor_log_prob(output_rep, sample)[head]
        return sampled_correction(
            logp.reshape(n, m),
            draws.start_logp.index_select(0, parent_index),
            draws.scores.index_select(0, parent_index),
        )

    def correction_fn(
        self, actor: torch.nn.Module, eta: Mapping[str, float] | None = None
    ) -> InitialCorrectionFn | None:
        """The minibatch hook for ``update_labor_policy``; None when no head is admitted."""
        eta = dict(self.eta if eta is None else eta)
        heads = [h for h in INITIAL_POLICY_HEADS if float(eta[h]) == 1.0 and h not in self.blind]
        if not heads:
            return None
        batch = self.batch

        def correction(parent_index: torch.Tensor) -> InitialCorrection:
            tensors = _rows(batch.initial_tensors, parent_index)
            output = actor(tensors)
            terms = self.head_terms(output, tensors, parent_index, heads)
            total = None
            sub_heads = {}
            for h in heads:
                term, split = terms[h]
                total = term if total is None else total + term
                if split is not None:
                    sub_heads[h] = split
            assert total is not None
            return InitialCorrection(total, sub_heads or None)

        return correction


def _actor_outputs(actor: torch.nn.Module, tensors: Mapping[str, torch.Tensor], n: int) -> dict:
    """Actor output on all rows (no grad), in row chunks."""
    parts: list[dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for start in range(0, n, _FORWARD_CHUNK):
            index = torch.arange(start, min(n, start + _FORWARD_CHUNK))
            parts.append({k: v.detach() for k, v in actor(_rows(tensors, index)).items()})
    out = {}
    for name in parts[0]:
        if name == "movement_codes":
            out[name] = parts[0][name]
        else:
            out[name] = torch.cat([part[name] for part in parts])
    return out


def _encode(payload: Mapping[str, Any], population: int) -> np.ndarray:
    return la.action_features(payload, population=int(population))


def _require_populations(populations: Sequence[int], n: int) -> list[int]:
    if len(populations) != n:
        raise ValueError("parent populations do not align with the batch")
    out = []
    for value in populations:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError("parent population must be a nonnegative integer")
        out.append(int(value))
    return out


def prepare_iteration(
    frozen: FrozenA,
    actor: torch.nn.Module,
    batch: LaborAttemptPolicyBatch,
    populations: Sequence[int],
    *,
    eta: Mapping[str, float],
    key: Sequence[int],
    draws: int = DEFAULT_DRAWS,
) -> AIteration:
    """A(s,a), the categorical counterfactual values and the sampled-head draws, once.

    ``actor`` must be the behaviour policy of ``batch`` (the iteration-start actor).
    ``key`` keys the draws (``train`` uses (optimizer_seed, iteration)).
    """
    from .labor_action_adapter import action_adapter_digest, categorical_counterfactuals

    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 2:
        raise ValueError("the sampled-head estimator needs at least two draws per row")
    if set(eta) != set(INITIAL_POLICY_HEADS) or any(
        float(v) not in (0.0, 1.0) for v in eta.values()
    ):
        raise ValueError(f"eta must name exactly {list(INITIAL_POLICY_HEADS)} with 0/1 values")
    batch.require_behaviour_head_logp()
    multipliers = batch.require_cost_multipliers()
    weights, _ = batch.row_personality_weights()
    n = batch.parent_count
    populations = _require_populations(populations, n)
    seconds: dict[str, float] = {}
    counts: dict[str, int] = {}
    clock = time.perf_counter()
    contexts = frozen.contexts(batch.initial_tensors, batch.initial_privileged_raw)
    seconds["contexts"] = time.perf_counter() - clock

    clock = time.perf_counter()
    labor, nonlabor = batch.initial_labor_sample, batch.initial_nonlabor_sample
    executed = np.stack(
        [
            _encode(schema12_initial_action(labor, nonlabor, i), populations[i])
            for i in range(n)
        ]
    )
    executed_features = torch.from_numpy(executed)
    if batch.initial_action_features is not None and not torch.equal(
        executed_features, batch.initial_action_features["features"]
    ):
        raise ValueError(
            "re-encoded executed actions differ from the rollout's recorded action features "
            "(population or encoder mismatch)"
        )
    roots = torch.arange(n)
    a_exec = combine_head_units(
        frozen.score(contexts, executed_features, roots), weights, multipliers
    )
    seconds["executed"] = time.perf_counter() - clock
    counts["executedScores"] = n

    blind = frozen.blind_heads()
    layout, adapter = la.action_layout_state(), action_adapter_digest()

    def score_candidates(features: torch.Tensor) -> torch.Tensor:  # [N, K, 349] -> [N, K]
        k = features.shape[1]
        flat = features.reshape(n * k, -1)
        raw = frozen.score(contexts, flat, roots.repeat_interleave(k))
        return combine_head_units(
            {name: value.reshape(n, k) for name, value in raw.items()}, weights, multipliers
        )

    clock = time.perf_counter()
    categorical: dict[str, torch.Tensor] = {}
    expected: dict[str, torch.Tensor] = {}
    output = _actor_outputs(actor, batch.initial_tensors, n)
    from .labor_model import _nonlabor_output

    movement, movement_mask, _, fission, _, _, gate, _ = _nonlabor_output(output)
    for head in ("movement", "fission"):
        if head in blind:
            continue
        values = score_candidates(
            categorical_counterfactuals(
                executed_features, head, expected_layout=layout, expected_adapter_digest=adapter
            )
        )
        categorical[head] = values
        counts[f"{head}Scores"] = int(values.numel())
    if "movement" in categorical:
        probs = torch.softmax(movement.masked_fill(~movement_mask, -torch.inf), dim=1)
        expected["movement"] = s2.categorical_expectation(probs, categorical["movement"])
    if "fission" in categorical:
        expected["fission"] = s2.categorical_expectation(
            torch.softmax(fission, dim=1), categorical["fission"]
        )
    if "trade_gate" not in blind:
        groups = len(contracts.TRADE_GROUPS)
        values = torch.stack(
            [
                score_candidates(
                    categorical_counterfactuals(
                        executed_features,
                        "trade_gate",
                        group=g,
                        expected_layout=layout,
                        expected_adapter_digest=adapter,
                    )
                )
                for g in range(groups)
            ],
            dim=1,
        )  # [N, G, 3]
        categorical["trade_gate"] = values
        probs = torch.softmax(gate, dim=2)
        expected["trade_gate"] = (probs * values).sum(dim=2)  # [N, G]
        counts["trade_gateScores"] = int(values.numel())
    seconds["categorical"] = time.perf_counter() - clock

    clock = time.perf_counter()
    sampled: dict[str, SampledDraws] = {}
    for head in SAMPLED_HEADS:
        if head in blind:
            continue
        sampled[head] = _draw_head(
            head, output, batch, populations, contexts, frozen, weights, multipliers,
            key=key, draws=draws,
        )
        expected[head] = sampled[head].scores.mean(dim=1)
        counts[f"{head}Scores"] = int(sampled[head].scores.numel())
    seconds["sampled"] = time.perf_counter() - clock
    for head in blind:
        expected[head] = a_exec.clone()  # A(x) = A(s, a) for every x: residual exactly 0
    return AIteration(
        {h: float(eta[h]) for h in INITIAL_POLICY_HEADS},
        batch.initial_advantage.detach(),
        a_exec.detach(),
        categorical,
        expected,
        sampled,
        blind,
        batch,
        counts,
        seconds,
    )


def _draw_head(
    head: str,
    output: Mapping[str, torch.Tensor],
    batch: LaborAttemptPolicyBatch,
    populations: Sequence[int],
    contexts: Mapping[str, torch.Tensor],
    frozen: FrozenA,
    weights: torch.Tensor,
    multipliers: Mapping[str, float],
    *,
    key: Sequence[int],
    draws: int,
) -> SampledDraws:
    """M keyed draws of ``head`` per row (other heads at their sampled values)."""
    n = batch.parent_count
    labor_head = head in LABOR_POLICY_HEADS
    prefix = "specialty" if head == "specialty" else "activity"
    replaced = (
        (f"{prefix}_allocation", f"{prefix}_latent")
        if labor_head
        else (("trade_value_latent",) if head == "trade_value" else ("trade_share_latent",))
    )
    executed = batch.initial_labor_sample if labor_head else batch.initial_nonlabor_sample
    spliced_rows: list[dict[str, torch.Tensor]] = []
    features = np.zeros((n, draws, la.ACTION_FEATURE_WIDTH), dtype=np.float32)
    for i in range(n):
        generator = draw_generator(key, i, head)
        local = torch.full((draws,), i, dtype=torch.long)
        output_rep = _rows(output, local)
        with torch.no_grad():
            if labor_head:
                masks = _rows({k: batch.initial_tensors[k] for k in _LABOR_MASK_KEYS}, local)
                drawn = sample_labor_policy(output_rep, masks, generator=generator)
            else:
                drawn = sample_nonlabor_policy(output_rep, generator=generator)
        spliced = _rows(executed, local)
        for name in replaced:
            spliced[name] = drawn[name].detach()
        spliced_rows.append(spliced)
        labor_rep = spliced if labor_head else _rows(batch.initial_labor_sample, local)
        nonlabor_rep = _rows(batch.initial_nonlabor_sample, local) if labor_head else spliced
        for m in range(draws):
            features[i, m] = _encode(
                schema12_initial_action(labor_rep, nonlabor_rep, m), populations[i]
            )
    sample: dict[str, torch.Tensor] = {}
    for name in spliced_rows[0]:
        if name == "movement_codes" and spliced_rows[0][name].ndim == 1:
            sample[name] = spliced_rows[0][name]
        else:
            sample[name] = torch.cat([row[name] for row in spliced_rows])
    local = torch.arange(n).repeat_interleave(draws)
    output_rep = _rows(output, local)
    with torch.no_grad():
        if labor_head:
            masks = _rows({k: batch.initial_tensors[k] for k in _LABOR_MASK_KEYS}, local)
            start = labor_log_prob(output_rep, sample, masks)[head]
        else:
            start = nonlabor_log_prob(output_rep, sample)[head]
    flat = torch.from_numpy(features.reshape(n * draws, -1))
    raw = frozen.score(contexts, flat, local)
    scores = combine_head_units(
        {name: value.reshape(n, draws) for name, value in raw.items()}, weights, multipliers
    )
    return SampledDraws(head, draws, sample, start.reshape(n, draws).detach(), scores.detach())


# ---------------------------------------------------------------------------
# D7: lagged admission from the measured per-head gradient variance
# ---------------------------------------------------------------------------


class _TraceAccumulator:
    """Streaming trace of the unbiased covariance of per-row gradient vectors."""

    def __init__(self) -> None:
        self.total: torch.Tensor | None = None
        self.square = 0.0
        self.rows = 0

    def add(self, gradient: torch.Tensor) -> None:
        g = gradient.detach().to(torch.float64)
        self.total = g.clone() if self.total is None else self.total + g
        self.square += float((g * g).sum())
        self.rows += 1

    def variance(self) -> float:
        if self.rows < 2 or self.total is None:
            raise ValueError("need at least two rows")
        mean_square = float((self.total * self.total).sum()) / self.rows
        return max(0.0, (self.square - mean_square) / (self.rows - 1))


def _flat_gradient(value: torch.Tensor, params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    grads = torch.autograd.grad(value, list(params), retain_graph=True, allow_unused=True)
    return torch.cat(
        [
            (g if g is not None else torch.zeros_like(p)).reshape(-1)
            for g, p in zip(grads, params, strict=True)
        ]
    )


def per_row_head_estimates(
    prepared: AIteration,
    actor: torch.nn.Module,
    row: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """For one parent row: per head the (V-only, control-variate) one-row objectives.

    V-only: ``log pi_h(a) * GAE``. Control variate (eta_h = 1): ``log pi_h(a) * c_h + E_h``
    (factored heads: ``sum_g log pi_g(a_g) c_g + sum_g E_g``). Their parameter gradients
    are head h's one-row policy-gradient estimates.
    """
    batch = prepared.batch
    index = torch.tensor([row], dtype=torch.long)
    tensors = _rows(batch.initial_tensors, index)
    output = actor(tensors)
    labor = labor_log_prob(output, _rows(batch.initial_labor_sample, index), tensors)
    nonlabor = nonlabor_log_prob(output, _rows(batch.initial_nonlabor_sample, index))
    logp = {
        **{h: labor[h] for h in LABOR_POLICY_HEADS},
        **{h: nonlabor[h] for h in NONLABOR_POLICY_HEADS},
    }
    gae = prepared.advantage.index_select(0, index)
    coefficients = prepared.coefficients({h: 1.0 for h in INITIAL_POLICY_HEADS})
    heads = [h for h in INITIAL_POLICY_HEADS if h not in prepared.blind]
    terms = prepared.head_terms(output, tensors, index, heads)
    out = {}
    for h in INITIAL_POLICY_HEADS:
        v_only = (logp[h] * gae).sum()
        if h in prepared.blind:
            out[h] = (v_only, v_only)
            continue
        term, split = terms[h]
        if split is not None:
            group_logp, group_coefficient = split
            score = (group_logp * group_coefficient).sum()
        else:
            score = (logp[h] * coefficients[h].index_select(0, index)).sum()
        out[h] = (v_only, score + term.sum())
    return out


def measure_d7(
    prepared: AIteration,
    actor: torch.nn.Module,
    *,
    rows: int = DEFAULT_D7_ROWS,
    threshold: float | None = None,
) -> tuple[dict[str, dict[str, Any]], float]:
    """Per policy head: V-only vs control-variate gradient variance and the D7 decision.

    Uses ``fixed_audit_indices(N, rows)`` rows of the prepared batch, one single-row actor
    forward each and two backward passes per non-blind head (one for A-blind heads, whose
    control-variate estimate is the V-only one exactly; they are never admitted).
    Returns (per-head measurements, seconds).
    """
    threshold = D7_MIN_VARIANCE_REDUCTION if threshold is None else float(threshold)
    params = [p for p in actor.parameters() if p.requires_grad]
    chosen = fixed_audit_indices(prepared.batch.parent_count, rows)
    if len(chosen) < 2:  # no variance from one row: nothing is admitted next iteration
        return {
            h: {
                "varianceVOnly": None,
                "varianceControlVariate": None,
                "reduction": None,
                "admit": False,
                "rows": len(chosen),
                "aBlind": h in prepared.blind,
                "threshold": threshold,
                "reason": "fewer than two rows: not measured",
            }
            for h in INITIAL_POLICY_HEADS
        }, 0.0
    v_acc = {h: _TraceAccumulator() for h in INITIAL_POLICY_HEADS}
    cv_acc = {h: _TraceAccumulator() for h in INITIAL_POLICY_HEADS}
    clock = time.perf_counter()
    for row in chosen.tolist():
        estimates = per_row_head_estimates(prepared, actor, int(row))
        for h, (v_only, cv) in estimates.items():
            v_grad = _flat_gradient(v_only, params)
            v_acc[h].add(v_grad)
            cv_acc[h].add(v_grad if h in prepared.blind else _flat_gradient(cv, params))
    seconds = time.perf_counter() - clock
    out = {}
    for h in INITIAL_POLICY_HEADS:
        var_v, var_cv = v_acc[h].variance(), cv_acc[h].variance()
        reduction = 1.0 - var_cv / var_v if var_v > 0 else 0.0
        out[h] = {
            "varianceVOnly": var_v,
            "varianceControlVariate": var_cv,
            "reduction": reduction,
            "admit": bool(h not in prepared.blind and var_v > 0 and reduction >= threshold),
            "rows": len(chosen),
            "aBlind": h in prepared.blind,
            "threshold": threshold,
        }
    return out, seconds


def eta_from_measurements(measurements: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    return {h: 1.0 if bool(measurements[h]["admit"]) else 0.0 for h in INITIAL_POLICY_HEADS}


def eta_for_iteration(
    state: Mapping[str, Any] | None,
    *,
    iteration: int,
    frozen: FrozenA,
) -> tuple[dict[str, float], str]:
    """The lagged eta: the decision measured on the previous iteration's batch.

    No stored state (the first gae iteration): eta = 0 on every head (measure only).
    A stored decision measured against another A/critic, or not on the immediately
    preceding iteration, is refused (an exact resume needs the matching state).
    """
    if state is None:
        return zero_eta(), "first-gae-iteration-measure-only"
    if not isinstance(state, Mapping) or state.get("schema") != A_STATE_SCHEMA:
        raise ValueError("checkpoint A control-variate state is malformed")
    if (
        state.get("criticDigest") != frozen.critic_digest
        or state.get("headDigest") != frozen.head_digest
    ):
        raise ValueError("checkpoint D7 state was measured with another frozen A/critic")
    if int(state.get("measuredIteration", -2)) != int(iteration) - 1:
        raise ValueError("checkpoint D7 state was not measured on the preceding iteration")
    eta = state.get("nextEta")
    if not isinstance(eta, Mapping) or set(eta) != set(INITIAL_POLICY_HEADS):
        raise ValueError("checkpoint D7 eta is malformed")
    return {h: float(eta[h]) for h in INITIAL_POLICY_HEADS}, "lagged-d7-from-previous-batch"


# ---------------------------------------------------------------------------
# B5: warm-up switch telemetry
# ---------------------------------------------------------------------------


def _world_calibration(
    pred: np.ndarray, realised: np.ndarray, world: np.ndarray
) -> dict[str, Any]:
    """World-balanced V calibration (``labor_a_stage1.calibration_metrics`` with one group
    per world): slope of realised on predicted after per-world demeaning, bias =
    mean(pred - realised), spread = per-world SD of realised, RMS over worlds."""
    per_world = {}
    for w in sorted(set(world.tolist())):
        sel = world == w
        p, y = pred[sel], realised[sel]
        dp, dy = p - p.mean(), y - y.mean()
        var = float(np.dot(dp, dp))
        per_world[int(w)] = {
            "rows": int(sel.sum()),
            "slope": None if var == 0 else float(np.dot(dp, dy) / var),
            "bias": float(np.mean(p - y)),
            "spread": float(np.std(y)),
        }
    slopes = [v["slope"] for v in per_world.values() if v["slope"] is not None]
    return {
        "slope": None if not slopes else float(np.mean(slopes)),
        "bias": float(np.mean([v["bias"] for v in per_world.values()])),
        "spread": float(np.sqrt(np.mean([v["spread"] ** 2 for v in per_world.values()]))),
        "perWorld": {str(w): v for w, v in per_world.items()},
    }


def warmup_report(
    online_values: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
    *,
    dry_run_kl: Sequence[float] | None,
) -> tuple[s2.WarmupReport | None, dict[str, Any]]:
    """Calibration and per-world TD(0) of the online V on the batch, per value component.

    ``online_values[name]`` [N]: the online V (raw units) at each parent's initial state.
    Realised returns are the discounted rollout returns from the per-decision rewards in
    ``decisionRows`` (``rewards``: physical task reward, feasibility initial + retry cost,
    settlement guard), with the rollout's frozen collect-time tail bootstrap for a living
    sequence (as GAE). TD(0) = r + gamma V(next) - V with the online V at the next decision
    and the frozen tail bootstrap after the last. Returns (None, reason) when the rollout
    predates the recorded rewards.
    """
    rows = metadata.get("decisionRows")
    components = (*LABOR_OBJECTIVES, *LABOR_COSTS)
    if not isinstance(rows, list) or any(
        not isinstance(r, Mapping) or not isinstance(r.get("rewards"), Mapping) for r in rows
    ):
        return None, {"status": "not measured: rollout predates per-decision rewards"}
    n = len(rows)
    if any(len(online_values[name]) != n for name in components):
        raise ValueError("online values do not align with the rollout decision rows")
    tails = {
        (int(entry["world"]), int(entry["camp"])): entry["values"]
        for entry in metadata.get("bootstrapValues", [])
    }
    sequences: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        sequences[(int(row["world"]), int(row["camp"]))].append(index)
    values = {name: online_values[name].detach().to(torch.float64).numpy() for name in components}
    realised = {name: np.zeros(n) for name in components}
    td = {name: np.zeros(n) for name in components}
    for key, indices in sequences.items():
        indices.sort(key=lambda i: int(rows[i]["turn"]))
        last = rows[indices[-1]]
        for name in components:
            following_return = 0.0
            following_value = 0.0
            if not bool(last["terminated"]):
                tail = tails.get(key)
                if tail is None or name not in tail:
                    return None, {"status": f"not measured: no tail bootstrap for {key}"}
                following_return = following_value = float(tail[name])
            for i in reversed(indices):
                row = rows[i]
                gamma = 0.0 if bool(row["terminated"]) else math.exp(
                    -float(row["years"]) / contracts.HORIZON_YEARS
                )
                reward = float(row["rewards"][name])
                realised[name][i] = reward + gamma * following_return
                td[name][i] = reward + gamma * following_value - values[name][i]
                following_return = realised[name][i]
                following_value = values[name][i]
    world = np.asarray([int(r["world"]) for r in rows], dtype=np.int64)
    detail: dict[str, Any] = {"status": "measured", "components": {}}
    slope, bias, spread, td_by_world = {}, {}, {}, {}
    for name in components:
        calibration = _world_calibration(values[name], realised[name], world)
        per_world = s2.per_world_td(world, td[name])
        slope[name], bias[name], spread[name] = (
            calibration["slope"], calibration["bias"], calibration["spread"],
        )
        td_by_world[name] = {w: v["mean"] for w, v in per_world.items()}
        detail["components"][name] = {
            **calibration,
            "events": calibration["spread"] > 0.0,
            "tdByWorld": {str(w): v for w, v in per_world.items()},
        }
    report = s2.WarmupReport(slope, bias, spread, td_by_world, dry_run_kl)
    detail["dryRunKL"] = "not measured" if dry_run_kl is None else list(dry_run_kl)
    return report, detail


def switch_state(state: Mapping[str, Any] | None) -> s2.WarmupSwitch:
    switch = s2.WarmupSwitch()
    if state is not None and isinstance(state.get("warmupSwitch"), Mapping):
        switch.phase = str(state["warmupSwitch"]["phase"])
        switch.streak = int(state["warmupSwitch"]["streak"])
    return switch


def json_number(value: Any) -> Any:
    """Plain JSON/weights-only-safe numbers (finite floats, None otherwise)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


__all__ = [
    "A_MODES",
    "A_STATE_KEY",
    "A_STATE_SCHEMA",
    "CATEGORICAL_HEADS",
    "D7_MIN_VARIANCE_REDUCTION",
    "DEFAULT_D7_ROWS",
    "DEFAULT_DRAWS",
    "DRAW_TAG",
    "FACTORED_HEADS",
    "SAMPLED_HEADS",
    "AIteration",
    "FrozenA",
    "combine_head_units",
    "draw_generator",
    "eta_for_iteration",
    "eta_from_measurements",
    "leave_one_out_centred",
    "load_frozen_a",
    "measure_d7",
    "per_row_head_estimates",
    "prepare_iteration",
    "sampled_correction",
    "switch_state",
    "warmup_report",
    "zero_eta",
]
