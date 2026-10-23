"""Narrow collection plumbing for the approved Gate 3/4 diagnostics.

This module does not load a checkpoint, choose a cohort, run the simulator, or perform
an optimizer step.  It provides the state-preserving fork boundary and return accounting
that a reviewed collector can use after those identities have been frozen.

The value contract mirrored here is PPO's ``(world, camp)`` sequence contract.  A camp's
future daughters are not silently added to its value target.  Reward rows must already
have been computed from the complete contemporaneous transition (including relatives)
with the checkpoint-bound reward implementation; filtering villages before lineage
reward construction would change the reward definition.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, isfinite
from types import TracebackType
from typing import Any, Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from . import contracts
from .independent_critic_scorecard import IndependentTarget
from .protocol import seeds as validate_seeds

Objective = Literal["welfare", "population_years", "trade_utility", "qol_years", "guard_cost"]
RewardSemantics = Literal[
    "raw-stationary",
    "history-conditioned-shaped",
    "unresolved-shaped",
]
DOTNET_INT32_MAX = 2_147_483_647
TailEstimateStatus = Literal[
    "identified",
    "unresolved-absolute-integrability",
    "unresolved-finite-variance",
    "unresolved-hard-cap",
    "unresolved-insufficient-draws",
]


class ForkBridge(Protocol):
    """The existing bridge operations used by :class:`ForkedAlternatives`."""

    def fork(self) -> tuple[int, dict[str, Any]]: ...

    def restore(self, fork: int) -> dict[str, Any]: ...

    def drop(self, fork: int) -> None: ...


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _step_identity(step: dict[str, Any]) -> tuple[int, str]:
    if "turn" not in step or not step.get("fingerprint"):
        raise ValueError("fork step requires turn and fingerprint")
    return int(step["turn"]), str(step["fingerprint"])


class ForkedAlternatives:
    """Keep every alternative on one exact in-process bridge fork.

    ``Model9Bridge.capture`` stores only the world.  It omits per-village bridge context,
    so it is not an exact substitute for ``fork``/``restore`` at a decision boundary.
    This scope restores before every alternative and once more before dropping the fork,
    leaving the live path at the original decision state even when branch code raises.
    """

    def __init__(self, bridge: ForkBridge) -> None:
        self._bridge = bridge
        self._fork: int | None = None
        self._identity: tuple[int, str] | None = None
        self._alternatives: set[str] = set()

    @property
    def decision_identity(self) -> tuple[int, str]:
        if self._identity is None:
            raise RuntimeError("fork scope has not been entered")
        return self._identity

    def __enter__(self) -> ForkedAlternatives:
        if self._fork is not None:
            raise RuntimeError("fork scope cannot be entered twice")
        fork, step = self._bridge.fork()
        self._fork = int(fork)
        try:
            self._identity = _step_identity(step)
        except BaseException:
            self._bridge.drop(self._fork)
            self._fork = None
            raise
        return self

    def begin(self, alternative_id: str) -> dict[str, Any]:
        """Restore the decision state and begin one uniquely named alternative."""
        if self._fork is None or self._identity is None:
            raise RuntimeError("fork scope has not been entered")
        if not alternative_id:
            raise ValueError("alternative_id must be nonempty")
        if alternative_id in self._alternatives:
            raise ValueError(f"duplicate alternative_id: {alternative_id}")
        step = self._bridge.restore(self._fork)
        if _step_identity(step) != self._identity:
            raise ValueError("bridge restore changed decision identity")
        self._alternatives.add(alternative_id)
        return step

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        if self._fork is None:
            return False
        cleanup_error: BaseException | None = None
        try:
            step = self._bridge.restore(self._fork)
            if self._identity is not None and _step_identity(step) != self._identity:
                raise ValueError("bridge cleanup restore changed decision identity")
        except BaseException as error:  # drop is still mandatory after a failed restore
            cleanup_error = error
        try:
            self._bridge.drop(self._fork)
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
            elif hasattr(cleanup_error, "add_note"):
                cleanup_error.add_note(f"fork drop also failed: {error!r}")
        finally:
            self._fork = None
        if cleanup_error is not None:
            if exc is not None and hasattr(exc, "add_note"):
                exc.add_note(f"fork cleanup failed: {cleanup_error!r}")
                return False
            raise cleanup_error
        return False


@dataclass(frozen=True)
class CampReward:
    """One already-derived reward on an opening camp's branch sequence."""

    camp_id: int
    step_index: int
    reward: float
    years: float
    terminated: bool

    def __post_init__(self) -> None:
        if isinstance(self.camp_id, bool):
            raise ValueError("camp_id must be an integer")
        if self.step_index < 0:
            raise ValueError("step_index must be nonnegative")
        _finite(self.reward, "reward")
        if _finite(self.years, "years") <= 0:
            raise ValueError("years must be positive")


@dataclass(frozen=True)
class CampPrefix:
    """Discounted sum over the camps present at the frozen decision state.

    ``live_tail_discounts`` has one entry for every opening camp still alive at the
    collection horizon.  Descendants may be needed when deriving contemporaneous lineage
    rewards, but they are not new value-sequence replicates and do not enter this sum.
    """

    prefix_return: float
    opening_camp_count: int
    terminated_camp_count: int
    live_tail_discounts: tuple[float, ...]
    collected_turns: int

    @property
    def terminal(self) -> bool:
        return not self.live_tail_discounts


@dataclass(frozen=True)
class CampOneStepValue:
    """One opening camp's immediate reward and same-camp successor value."""

    camp_id: int
    immediate_reward: float
    years: float
    successor_value: float
    terminated: bool

    def __post_init__(self) -> None:
        _finite(self.immediate_reward, "immediate_reward")
        if _finite(self.years, "years") <= 0:
            raise ValueError("years must be positive")
        successor = _finite(self.successor_value, "successor_value")
        if self.terminated and successor != 0:
            raise ValueError("terminated camp must have zero successor value")


@dataclass(frozen=True)
class RankingPrediction:
    """World-cluster aggregate of ``r + gamma * V(successor)`` for one action."""

    immediate_reward: float
    gamma: float
    successor_value: float
    opening_camp_count: int

    @property
    def value(self) -> float:
        return float(self.immediate_reward + self.gamma * self.successor_value)


def aggregate_ranking_prediction(
    rows: Sequence[CampOneStepValue],
    opening_camps: Sequence[int],
    *,
    horizon_years: float = contracts.HORIZON_YEARS,
) -> RankingPrediction:
    """Build an action-ranking prediction from matched same-camp successors.

    A value evaluated at the shared pre-action state cannot rank alternatives.  This
    function instead aggregates each alternative's realized immediate reward and the
    tested critic's value at that alternative's successor.  Newly founded camps are not
    successors of an opening camp under PPO's camp-sequence contract.
    """
    camps = tuple(int(camp) for camp in opening_camps)
    if not camps or len(camps) != len(set(camps)):
        raise ValueError("opening_camps must be nonempty and unique")
    selected: dict[int, CampOneStepValue] = {}
    for row in rows:
        if row.camp_id in camps:
            if row.camp_id in selected:
                raise ValueError(f"duplicate one-step row for opening camp {row.camp_id}")
            selected[row.camp_id] = row
    missing = [camp for camp in camps if camp not in selected]
    if missing:
        raise ValueError(f"missing one-step rows for opening camps: {missing}")
    years = {float(selected[camp].years) for camp in camps}
    if len(years) != 1:
        raise ValueError("one world transition must use one elapsed-years discount")
    horizon_years = _finite(horizon_years, "horizon_years")
    if horizon_years <= 0:
        raise ValueError("horizon_years must be positive")
    gamma = exp(-years.pop() / horizon_years)
    return RankingPrediction(
        immediate_reward=float(sum(selected[camp].immediate_reward for camp in camps)),
        gamma=float(gamma),
        successor_value=float(sum(selected[camp].successor_value for camp in camps)),
        opening_camp_count=len(camps),
    )


def aggregate_camp_prefix(
    rows: Sequence[CampReward],
    opening_camps: Sequence[int],
    collected_turns: int,
    *,
    horizon_years: float = contracts.HORIZON_YEARS,
) -> CampPrefix:
    """Aggregate complete camp sequences without counting villages as independent units."""
    if collected_turns <= 0:
        raise ValueError("collected_turns must be positive")
    horizon_years = _finite(horizon_years, "horizon_years")
    if horizon_years <= 0:
        raise ValueError("horizon_years must be positive")
    camps = tuple(int(camp) for camp in opening_camps)
    if not camps or len(camps) != len(set(camps)):
        raise ValueError("opening_camps must be nonempty and unique")

    by_camp: dict[int, list[CampReward]] = {camp: [] for camp in camps}
    for row in rows:
        if row.camp_id in by_camp:
            by_camp[row.camp_id].append(row)

    total = 0.0
    terminated = 0
    live_tail_discounts: list[float] = []
    for camp in camps:
        sequence = sorted(by_camp[camp], key=lambda row: row.step_index)
        if not sequence or sequence[0].step_index != 0:
            raise ValueError(f"opening camp {camp} lacks its first branch outcome")
        indices = [row.step_index for row in sequence]
        if len(indices) != len(set(indices)) or indices != list(range(len(indices))):
            raise ValueError(f"opening camp {camp} has duplicate or discontinuous outcomes")
        ended = [index for index, row in enumerate(sequence) if row.terminated]
        if ended and (len(ended) != 1 or ended[0] != len(sequence) - 1):
            raise ValueError(f"opening camp {camp} has outcomes after termination")
        if not ended and len(sequence) != collected_turns:
            raise ValueError(f"living opening camp {camp} has an incomplete prefix")
        if len(sequence) > collected_turns:
            raise ValueError(f"opening camp {camp} exceeds the declared collection horizon")

        discount = 1.0
        for row in sequence:
            total += discount * row.reward
            discount *= exp(-row.years / horizon_years)
        if ended:
            terminated += 1
        else:
            live_tail_discounts.append(discount)

    return CampPrefix(
        prefix_return=float(total),
        opening_camp_count=len(camps),
        terminated_camp_count=terminated,
        live_tail_discounts=tuple(live_tail_discounts),
        collected_turns=collected_turns,
    )


def close_prefix_target(
    prefix: CampPrefix,
    objective: Objective,
    provenance_id: str,
    *,
    shaped: bool,
    turn_years: float = contracts.TURN_YEARS,
    horizon_years: float = contracts.HORIZON_YEARS,
) -> IndependentTarget:
    """Close a realized prefix only as far as source-backed bounds permit.

    Terminal camp sequences are exact.  For unshaped PPO component rewards, welfare and
    population-years are nonnegative and therefore give a lower tail bound of zero.
    QoL is clamped to ``[0, 1]`` by the simulator, so a living camp contributes at most
    ``turn_years`` QoL-years per fixed turn.  Guard cost is an indicator and contributes at
    most one per turn.  Trade utility is signed.  Shaped rewards receive no implicit
    nonnegativity assumption: without an explicit debit bound their tail remains unresolved.
    """
    if objective not in {"welfare", "population_years", "trade_utility", "qol_years", "guard_cost"}:
        raise ValueError(f"unsupported objective: {objective}")
    if prefix.terminal:
        return IndependentTarget(prefix.prefix_return, "terminal", provenance_id)
    if shaped or objective == "trade_utility":
        return IndependentTarget(prefix.prefix_return, "unresolved", provenance_id)
    if objective in {"welfare", "population_years"}:
        return IndependentTarget(
            prefix.prefix_return,
            "bounded",
            provenance_id,
            tail_lower=0.0,
        )

    turn_years = _finite(turn_years, "turn_years")
    horizon_years = _finite(horizon_years, "horizon_years")
    if turn_years <= 0 or horizon_years <= 0:
        raise ValueError("turn_years and horizon_years must be positive")
    gamma = exp(-turn_years / horizon_years)
    maximum_per_turn = turn_years if objective == "qol_years" else 1.0
    tail_upper = sum(
        discount * maximum_per_turn / (1.0 - gamma) for discount in prefix.live_tail_discounts
    )
    return IndependentTarget(
        prefix.prefix_return,
        "bounded",
        provenance_id,
        tail_lower=0.0,
        tail_upper=float(tail_upper),
    )


@dataclass(frozen=True)
class TailAssumptions:
    """Source-backed conditions required by a randomized infinite-tail estimate.

    Absolute integrability is needed to exchange the expectation over the stopping
    draw with the infinite sum.  A finite second moment is separately needed for the
    reported Monte Carlo standard error.  A finite observed prefix alone establishes
    neither condition.  Raw camp welfare/population can instead use the simulator's
    signed-Int32 population ceiling; signed trade and shaped rewards need separate proofs.
    """

    absolute_integrability_proven: bool
    finite_variance_proven: bool
    provenance_id: str

    def __post_init__(self) -> None:
        if not self.provenance_id:
            raise ValueError("tail assumptions require provenance_id")


def raw_camp_reward_absolute_bound(
    objective: Objective,
    opening_camp_count: int,
    *,
    turn_years: float = contracts.TURN_YEARS,
    population_max: int = DOTNET_INT32_MAX,
) -> float:
    """Bound one aggregate raw reward term for the fixed opening camps.

    The pinned simulator stores ``Camp.Population`` and the emitted prior population as
    signed C# ``int`` values.  Current ``lineage_rewards`` clamps prior population below
    at zero; its lineage QoL/alive average is in ``[0, 1]``.  Thus each opening camp's
    raw welfare or population-years row is at most ``int.MaxValue * years``; raw QoL-
    years is at most ``years`` and guard cost at most one.  The opening-camp set is fixed
    at the decision state: daughters do not increase this sequence bound.

    This does not bound signed trade utility or any shaped reward.  Collection still
    has to bind the exact simulator/configuration and verify elapsed years on every row.
    """
    if (
        isinstance(opening_camp_count, bool)
        or int(opening_camp_count) != opening_camp_count
        or opening_camp_count <= 0
    ):
        raise ValueError("opening_camp_count must be a positive integer")
    if (
        isinstance(population_max, bool)
        or int(population_max) != population_max
        or not 0 < population_max <= DOTNET_INT32_MAX
    ):
        raise ValueError("population_max must be a positive signed Int32 bound")
    years = _finite(turn_years, "turn_years")
    if years <= 0:
        raise ValueError("turn_years must be positive")
    if objective in {"welfare", "population_years"}:
        bound = opening_camp_count * population_max * years
    elif objective == "qol_years":
        bound = opening_camp_count * years
    elif objective == "guard_cost":
        bound = float(opening_camp_count)
    elif objective == "trade_utility":
        raise ValueError("signed trade utility has no source-backed bound here")
    else:
        raise ValueError(f"unsupported objective: {objective}")
    return float(bound)


def shaped_welfare_reward_absolute_bound(
    raw_lineage_reward_bound: float,
    native_welfare_bound: float,
    *,
    mix_change_cost: float,
    break_cost: float,
    commitments_active: bool,
) -> float:
    """Bound the shaped scalar-welfare reward used by recovered PPO.

    Recovered ``ppo.iteration`` subtracts ``welfare_now`` times a mix-change
    total-variation cost and, only for a commitments stage, a voluntary-break cost.
    Total variation between probability vectors is at most one and the break term is
    an indicator.  This bound establishes magnitude only; it does not turn the
    history- and rollout-boundary-dependent shaping rule into a stationary state reward.
    """
    raw_bound = _finite(raw_lineage_reward_bound, "raw_lineage_reward_bound")
    welfare_bound = _finite(native_welfare_bound, "native_welfare_bound")
    mix_cost = _finite(mix_change_cost, "mix_change_cost")
    hold_break_cost = _finite(break_cost, "break_cost")
    if raw_bound < 0 or welfare_bound < 0:
        raise ValueError("reward bounds must be nonnegative")
    debit_share = abs(mix_cost)
    if commitments_active:
        debit_share += abs(hold_break_cost)
    return float(raw_bound + debit_share * welfare_bound)


def shaped_nonnegative_component_absolute_bound(
    component_bound: float,
    *,
    mix_change_cost: float,
    break_cost: float,
    commitments_active: bool,
) -> float:
    """Bound a nonnegative component after recovered ``shape_components``.

    If ``0 <= component <= B``, the shaped value is
    ``component - share * component``.  With mix total variation and the optional
    break indicator each bounded by one, its absolute value is at most
    ``(1 + |c_mix| + I_commitments |c_break|) B``.
    """
    bound = _finite(component_bound, "component_bound")
    mix_cost = _finite(mix_change_cost, "mix_change_cost")
    hold_break_cost = _finite(break_cost, "break_cost")
    if bound < 0:
        raise ValueError("component_bound must be nonnegative")
    debit_share = abs(mix_cost)
    if commitments_active:
        debit_share += abs(hold_break_cost)
    return float((1.0 + debit_share) * bound)


def _probability_mix(values: Sequence[float], name: str) -> NDArray[np.float64]:
    mix = np.asarray(tuple(values), dtype=np.float64)
    if mix.ndim != 1 or mix.size < 2:
        raise ValueError(f"{name} must be a one-dimensional probability vector")
    if not np.all(np.isfinite(mix)) or np.any(mix < 0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    if not np.isclose(float(mix.sum()), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError(f"{name} must sum to one")
    return mix


def shaped_welfare_reward_at_fork(
    raw_lineage_reward: float,
    native_welfare: float,
    current_policy_mix: Sequence[float],
    *,
    previous_policy_mix: Sequence[float] | None,
    sequence_initial_decision: bool,
    mix_change_cost: float,
    voluntary_break: bool = False,
    break_cost: float = 0.0,
) -> float:
    """Reproduce one recovered shaped-welfare row without resetting at a fork.

    Recovered ``mix_changes`` assigns zero only to the first row of each collected camp
    sequence.  This legitimately includes a newly born camp's first decision as well as
    a camp at a fresh world reset.  An arbitrary later-state fork is not equivalent.
    Therefore a non-initial fork must carry the exact previous sampled policy mix from
    the main path.  Simulator state or a bridge fork alone is not silently treated as
    that latent policy history.
    """
    raw = _finite(raw_lineage_reward, "raw_lineage_reward")
    welfare = max(0.0, _finite(native_welfare, "native_welfare"))
    current = _probability_mix(current_policy_mix, "current_policy_mix")
    mix_cost = _finite(mix_change_cost, "mix_change_cost")
    hold_break_cost = _finite(break_cost, "break_cost")
    if previous_policy_mix is None:
        if not sequence_initial_decision:
            raise ValueError(
                "a non-initial fork requires the exact previous sampled policy mix; "
                "do not reset mix-change shaping at the fork"
            )
        change = 0.0
    else:
        if sequence_initial_decision:
            raise ValueError("a sequence-initial decision may not carry a previous policy mix")
        previous = _probability_mix(previous_policy_mix, "previous_policy_mix")
        if previous.shape != current.shape:
            raise ValueError("current and previous policy mixes must have equal width")
        change = 0.5 * float(np.abs(current - previous).sum())
    share = mix_cost * change + hold_break_cost * float(bool(voluntary_break))
    return float(raw - share * welfare)


def bounded_geometric_tail_assumptions(
    *,
    per_turn_absolute_bound: float,
    discount: float,
    survival_probability: float,
    provenance_id: str,
) -> TailAssumptions:
    """Validate a sufficient bounded-geometric proof for the roulette estimator.

    If ``|a_t| <= bound * discount**t``, absolute integrability follows for
    ``discount < 1`` and a finite second moment follows when roulette survival
    ``rho > discount**2``.  Raw camp welfare, population-years, QoL-years, and guard
    cost can use :func:`raw_camp_reward_absolute_bound`.  Signed trade and shaped
    rewards need separate source-backed bounds.
    """
    bound = _finite(per_turn_absolute_bound, "per_turn_absolute_bound")
    gamma = _finite(discount, "discount")
    rho = _finite(survival_probability, "survival_probability")
    if bound < 0:
        raise ValueError("per_turn_absolute_bound must be nonnegative")
    if not 0 <= gamma < 1:
        raise ValueError("discount must be in [0, 1)")
    if not 0 < rho < 1:
        raise ValueError("survival_probability must be strictly between zero and one")
    if rho <= gamma**2:
        raise ValueError("finite-variance proof requires survival_probability > discount**2")
    return TailAssumptions(True, True, provenance_id)


def geometric_stop_indices(
    uniforms: Sequence[float], survival_probability: float
) -> tuple[int, ...]:
    """Transform frozen uniforms into inclusive geometric stopping indices.

    For returned ``N``, ``P(N >= t) = survival_probability ** t``.  This explicit
    transform makes the stopping schedule auditable independently of a random-number
    generator implementation.
    """
    rho = _finite(survival_probability, "survival_probability")
    if not 0 < rho < 1:
        raise ValueError("survival_probability must be strictly between zero and one")
    values = np.asarray(tuple(uniforms), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("uniforms must be a nonempty one-dimensional sequence")
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values >= 1):
        raise ValueError("uniforms must be finite values in [0, 1)")
    stops = np.floor(np.log1p(-values) / np.log(rho)).astype(np.int64)
    return tuple(int(stop) for stop in stops)


@dataclass(frozen=True)
class FrozenGeometricSchedule:
    """A pre-result Russian-roulette stopping schedule.

    ``stop_indices`` are inclusive.  If even one draw requires more than the hard cap,
    the schedule is retained as evidence but is not executable; callers must not redraw,
    discard that draw, or truncate it to the cap.
    """

    schedule_id: str
    survival_probability: float
    stop_indices: tuple[int, ...]
    hard_cap_terms: int
    rng_seed: int | None = None
    generator: str = "predeclared-explicit"
    frozen_before_collection: bool = True

    def __post_init__(self) -> None:
        if not self.schedule_id:
            raise ValueError("schedule_id must be nonempty")
        rho = _finite(self.survival_probability, "survival_probability")
        if not 0 < rho < 1:
            raise ValueError("survival_probability must be strictly between zero and one")
        if not self.stop_indices or any(
            isinstance(stop, bool) or int(stop) != stop or stop < 0 for stop in self.stop_indices
        ):
            raise ValueError("stop_indices must be nonempty nonnegative integers")
        if (
            isinstance(self.hard_cap_terms, bool)
            or int(self.hard_cap_terms) != self.hard_cap_terms
            or self.hard_cap_terms <= 0
        ):
            raise ValueError("hard_cap_terms must be a positive integer")
        if not self.frozen_before_collection:
            raise ValueError("stopping schedule must be frozen before collection")
        if self.rng_seed is not None and (
            isinstance(self.rng_seed, bool) or int(self.rng_seed) != self.rng_seed
        ):
            raise ValueError("rng_seed must be an integer when present")
        if not self.generator:
            raise ValueError("generator must be nonempty")

    @property
    def draw_count(self) -> int:
        return len(self.stop_indices)

    @property
    def required_terms(self) -> int:
        return max(self.stop_indices) + 1

    @property
    def exceeds_hard_cap(self) -> bool:
        return self.required_terms > self.hard_cap_terms

    @property
    def expected_terms_per_draw(self) -> float:
        return float(1.0 / (1.0 - self.survival_probability))


def freeze_geometric_schedule(
    *,
    schedule_id: str,
    rng_seed: int,
    draw_count: int,
    survival_probability: float,
    hard_cap_terms: int,
) -> FrozenGeometricSchedule:
    """Freeze a PCG64 stopping schedule without interacting with simulator RNG."""
    if isinstance(rng_seed, bool) or int(rng_seed) != rng_seed:
        raise ValueError("rng_seed must be an integer")
    if isinstance(draw_count, bool) or int(draw_count) != draw_count or draw_count <= 0:
        raise ValueError("draw_count must be a positive integer")
    generator = np.random.Generator(np.random.PCG64(int(rng_seed)))
    uniforms = generator.random(int(draw_count))
    return FrozenGeometricSchedule(
        schedule_id=schedule_id,
        survival_probability=survival_probability,
        stop_indices=geometric_stop_indices(
            tuple(float(value) for value in uniforms), survival_probability
        ),
        hard_cap_terms=hard_cap_terms,
        rng_seed=int(rng_seed),
        generator="numpy.random.PCG64",
    )


@dataclass(frozen=True)
class RandomizedTailEstimate:
    """A tail estimate and its deliberately narrow uncertainty scope."""

    status: TailEstimateStatus
    schedule_id: str
    estimate: float | None = None
    standard_error: float | None = None
    draw_estimates: tuple[float, ...] = ()
    standard_error_scope: str | None = None
    stopping_randomization_contribution_to_standard_error: float | None = None
    estimate_kind: Literal["tail", "full-return"] = "tail"

    def __post_init__(self) -> None:
        if self.status not in {
            "identified",
            "unresolved-absolute-integrability",
            "unresolved-finite-variance",
            "unresolved-hard-cap",
            "unresolved-insufficient-draws",
        }:
            raise ValueError(f"unsupported randomized-tail status: {self.status}")
        if not self.schedule_id:
            raise ValueError("randomized tail estimate requires schedule_id")
        if self.estimate_kind not in {"tail", "full-return"}:
            raise ValueError(f"unsupported randomized estimate kind: {self.estimate_kind}")
        if self.status == "identified":
            if self.estimate is None or self.standard_error is None:
                raise ValueError("identified tail requires estimate and standard error")
            _finite(self.estimate, "tail estimate")
            if _finite(self.standard_error, "tail standard error") < 0:
                raise ValueError("tail standard error must be nonnegative")
            if len(self.draw_estimates) < 2:
                raise ValueError("identified tail requires at least two Monte Carlo values")
            for value in self.draw_estimates:
                _finite(value, "tail Monte Carlo value")
            if not self.standard_error_scope:
                raise ValueError("identified tail requires standard_error_scope")
            contribution = self.stopping_randomization_contribution_to_standard_error
            if contribution is None:
                raise ValueError(
                    "identified tail requires its stopping-randomization SE contribution"
                )
            if _finite(contribution, "stopping-randomization SE contribution") < 0:
                raise ValueError("stopping-randomization SE contribution must be nonnegative")
        elif (
            any(
                value is not None
                for value in (
                    self.estimate,
                    self.standard_error,
                    self.standard_error_scope,
                    self.stopping_randomization_contribution_to_standard_error,
                )
            )
            or self.draw_estimates
        ):
            raise ValueError("unresolved tail may not carry point estimates")

    @property
    def identified(self) -> bool:
        return self.status == "identified"


def _unresolved_tail(
    status: TailEstimateStatus, schedule: FrozenGeometricSchedule
) -> RandomizedTailEstimate:
    return RandomizedTailEstimate(status=status, schedule_id=schedule.schedule_id)


def estimate_randomized_tail(
    discounted_terms: Sequence[float],
    schedule: FrozenGeometricSchedule,
    assumptions: TailAssumptions,
) -> RandomizedTailEstimate:
    """Estimate an infinite continuation tail without a tested-critic bootstrap.

    ``discounted_terms[t]`` must already include every discount from the original
    frozen decision state through tail index ``t``.  One collected continuation path is
    reused for all frozen stopping draws.  The returned standard error is therefore only
    stopping-randomization Monte Carlo error conditional on that observed path; it is not
    action/simulator-path uncertainty and not finite-world sampling uncertainty.
    """
    if not assumptions.absolute_integrability_proven:
        return _unresolved_tail("unresolved-absolute-integrability", schedule)
    if not assumptions.finite_variance_proven:
        return _unresolved_tail("unresolved-finite-variance", schedule)
    if schedule.exceeds_hard_cap:
        return _unresolved_tail("unresolved-hard-cap", schedule)
    if schedule.draw_count < 2:
        return _unresolved_tail("unresolved-insufficient-draws", schedule)

    terms = tuple(_finite(term, "discounted tail term") for term in discounted_terms)
    if len(terms) < schedule.required_terms:
        raise ValueError(
            "discounted_terms do not cover the frozen stopping schedule: "
            f"need {schedule.required_terms}, received {len(terms)}"
        )
    rho = schedule.survival_probability
    draws = tuple(
        float(sum(terms[index] / rho**index for index in range(stop + 1)))
        for stop in schedule.stop_indices
    )
    values = np.asarray(draws, dtype=np.float64)
    estimate = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(values.size))
    if not isfinite(estimate) or not isfinite(standard_error):
        raise ValueError("randomized tail estimate is nonfinite")
    return RandomizedTailEstimate(
        status="identified",
        schedule_id=schedule.schedule_id,
        estimate=estimate,
        standard_error=standard_error,
        draw_estimates=draws,
        standard_error_scope=(
            "stopping-randomization Monte Carlo conditional on one observed "
            "continuation path; excludes action/simulator-path and world uncertainty"
        ),
        stopping_randomization_contribution_to_standard_error=standard_error,
    )


def estimate_paired_tail_difference(
    first_discounted_terms: Sequence[float],
    second_discounted_terms: Sequence[float],
    schedule: FrozenGeometricSchedule,
    assumptions: TailAssumptions,
) -> RandomizedTailEstimate:
    """Estimate ``first - second`` with the same stopping draw for both alternatives."""
    if not assumptions.absolute_integrability_proven:
        return _unresolved_tail("unresolved-absolute-integrability", schedule)
    if not assumptions.finite_variance_proven:
        return _unresolved_tail("unresolved-finite-variance", schedule)
    if schedule.exceeds_hard_cap:
        return _unresolved_tail("unresolved-hard-cap", schedule)
    required = schedule.required_terms
    if len(first_discounted_terms) < required or len(second_discounted_terms) < required:
        raise ValueError("paired discounted terms do not cover the frozen schedule")
    paired = tuple(
        _finite(first_discounted_terms[index], "first discounted tail term")
        - _finite(second_discounted_terms[index], "second discounted tail term")
        for index in range(required)
    )
    result = estimate_randomized_tail(paired, schedule, assumptions)
    if not result.identified:
        return result
    return RandomizedTailEstimate(
        status=result.status,
        schedule_id=result.schedule_id,
        estimate=result.estimate,
        standard_error=result.standard_error,
        draw_estimates=result.draw_estimates,
        standard_error_scope=(
            "paired stopping-randomization Monte Carlo conditional on paired observed "
            "continuation paths; excludes action/simulator-path and world uncertainty"
        ),
        stopping_randomization_contribution_to_standard_error=(
            result.stopping_randomization_contribution_to_standard_error
        ),
    )


def combine_independent_path_returns(
    fixed_prefix_returns: Sequence[float],
    tail_estimates: Sequence[RandomizedTailEstimate],
    provenance_id: str,
) -> RandomizedTailEstimate:
    """Average complete stochastic-path returns within one world.

    Every action/simulator path has its own realized fixed prefix and randomized tail.
    Adding them before cross-path aggregation preserves their covariance; adding one
    shared prefix after averaging tails is invalid.  Independent paths need independently
    frozen stopping schedules.  Worlds, states, camps, and stopping draws are not promoted
    to independent units.
    """
    if not provenance_id:
        raise ValueError("path combination requires provenance_id")
    prefixes = tuple(_finite(value, "fixed prefix return") for value in fixed_prefix_returns)
    estimates = tuple(tail_estimates)
    if not estimates:
        raise ValueError("tail_estimates must be nonempty")
    if len(prefixes) != len(estimates):
        raise ValueError("every tail estimate requires its own fixed prefix return")
    if any(item.estimate_kind != "tail" for item in estimates):
        raise ValueError("path combination inputs must be tail estimates")
    unresolved = next((item for item in estimates if not item.identified), None)
    if unresolved is not None:
        return RandomizedTailEstimate(
            status=unresolved.status,
            schedule_id=provenance_id,
            estimate_kind="full-return",
        )
    if len(estimates) < 2:
        raise ValueError(
            "policy-expectation estimation needs at least two independent action paths; "
            "a single deterministic path belongs in the intervention pane"
        )
    schedule_ids = [item.schedule_id for item in estimates]
    if len(schedule_ids) != len(set(schedule_ids)):
        raise ValueError("independent action paths require distinct frozen stopping schedules")
    assert all(item.estimate is not None for item in estimates)
    values = np.asarray(
        [
            prefix + cast(float, item.estimate)
            for prefix, item in zip(prefixes, estimates, strict=True)
        ],
        dtype=np.float64,
    )
    stopping_errors = np.asarray(
        [item.stopping_randomization_contribution_to_standard_error for item in estimates],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("path estimates must be finite")
    stopping_contribution = float(np.sqrt(np.sum(stopping_errors**2)) / len(estimates))
    return RandomizedTailEstimate(
        status="identified",
        schedule_id=provenance_id,
        estimate=float(values.mean()),
        standard_error=float(values.std(ddof=1) / np.sqrt(values.size)),
        draw_estimates=tuple(float(value) for value in values),
        standard_error_scope=(
            "complete fixed-prefix-plus-tail action/simulator paths, including their "
            "covariance and stopping Monte Carlo, conditional on one world; cross-world "
            "inference must cluster by world"
        ),
        stopping_randomization_contribution_to_standard_error=stopping_contribution,
        estimate_kind="full-return",
    )


def target_from_randomized_tail(
    prefix_return: float,
    tail: RandomizedTailEstimate,
    provenance_id: str,
) -> IndependentTarget:
    """Convert only an identified critic-independent tail into a point target."""
    prefix_return = _finite(prefix_return, "prefix_return")
    if not provenance_id:
        raise ValueError("target requires provenance_id")
    if tail.estimate_kind != "tail":
        raise ValueError("target_from_randomized_tail requires a tail estimate")
    if not tail.identified:
        return IndependentTarget(prefix_return, "unresolved", provenance_id)
    assert tail.estimate is not None and tail.standard_error is not None
    return IndependentTarget(
        prefix_return,
        "independent-estimate",
        provenance_id,
        tail_estimate=tail.estimate,
        tail_standard_error=tail.standard_error,
    )


@dataclass(frozen=True)
class ContinuationRegimeBinding:
    """Minimum semantic binding for a collected continuation pane.

    Only the sampled training-policy pane with a stationary raw reward can make an
    unconditional full-value calibration claim.  Recovered shaping depends on prior
    sampled policy mix and rollout-boundary treatment.  Such a pane is explicitly
    history-conditioned, never silently relabelled as a stationary state value.
    Deterministic continuations are separately named interventions.
    Hashes identify checkpoint, metadata, normalizer, simulator, bridge, configuration,
    producer, and reward bytes.  ``checkpoint_resolution_id`` must name the reviewed
    raw load/migration/default receipt; an implicit current ``upgrade()`` is inadmissible.
    """

    pane: Literal["sampled-training-policy", "deterministic-intervention"]
    actor_checkpoint_sha256: str
    critic_checkpoint_sha256: str
    checkpoint_metadata_sha256: str
    normalizer_sha256: str
    simulator_sha256: str
    bridge_sha256: str
    configuration_sha256: str
    producer_sha256: str
    reward_contract_sha256: str
    checkpoint_resolution_id: str
    critic_return_contract_id: str
    stage_id: str
    personality_id: str
    shaping_id: str
    reward_semantics: RewardSemantics
    action_rng_schedule_id: str | None = None
    simulator_rng_schedule_id: str | None = None
    exploration_contract_id: str | None = None
    log_std_floor: float | None = None
    shaping_history_id: str | None = None
    rollout_boundary_rule_id: str | None = None

    def __post_init__(self) -> None:
        required = {
            "actor_checkpoint_sha256": self.actor_checkpoint_sha256,
            "critic_checkpoint_sha256": self.critic_checkpoint_sha256,
            "checkpoint_metadata_sha256": self.checkpoint_metadata_sha256,
            "normalizer_sha256": self.normalizer_sha256,
            "simulator_sha256": self.simulator_sha256,
            "bridge_sha256": self.bridge_sha256,
            "configuration_sha256": self.configuration_sha256,
            "producer_sha256": self.producer_sha256,
            "reward_contract_sha256": self.reward_contract_sha256,
            "checkpoint_resolution_id": self.checkpoint_resolution_id,
            "critic_return_contract_id": self.critic_return_contract_id,
            "stage_id": self.stage_id,
            "personality_id": self.personality_id,
            "shaping_id": self.shaping_id,
        }
        empty = [name for name, value in required.items() if not value]
        if empty:
            raise ValueError(f"continuation binding has empty fields: {empty}")
        for name in (
            "actor_checkpoint_sha256",
            "critic_checkpoint_sha256",
            "checkpoint_metadata_sha256",
            "normalizer_sha256",
            "simulator_sha256",
            "bridge_sha256",
            "configuration_sha256",
            "producer_sha256",
            "reward_contract_sha256",
        ):
            digest = str(required[name])
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not self.simulator_rng_schedule_id:
            raise ValueError("continuation binding requires a simulator RNG schedule")
        if self.reward_semantics == "raw-stationary":
            if self.shaping_history_id is not None or self.rollout_boundary_rule_id is not None:
                raise ValueError("raw reward semantics may not carry shaping-history fields")
        elif self.reward_semantics == "history-conditioned-shaped":
            if not self.shaping_history_id or not self.rollout_boundary_rule_id:
                raise ValueError(
                    "history-conditioned shaping requires prior-policy history and "
                    "rollout-boundary rule identifiers"
                )
        elif self.reward_semantics != "unresolved-shaped":
            raise ValueError(f"unsupported reward semantics: {self.reward_semantics}")
        if self.pane == "sampled-training-policy":
            if not self.action_rng_schedule_id:
                raise ValueError(
                    "sampled training-policy pane requires action and simulator RNG schedules"
                )
            if not self.exploration_contract_id or self.log_std_floor is None:
                raise ValueError(
                    "sampled training-policy pane requires exact exploration contract "
                    "and log_std_floor"
                )
            _finite(self.log_std_floor, "log_std_floor")
        elif self.pane == "deterministic-intervention":
            if self.action_rng_schedule_id is not None or self.exploration_contract_id is not None:
                raise ValueError(
                    "deterministic intervention may not masquerade as a sampled actor path"
                )
        else:
            raise ValueError(f"unsupported continuation pane: {self.pane}")

    @property
    def supports_full_critic_calibration(self) -> bool:
        return self.pane == "sampled-training-policy" and self.reward_semantics == "raw-stationary"

    @property
    def supports_history_conditioned_critic_diagnostic(self) -> bool:
        return (
            self.pane == "sampled-training-policy"
            and self.reward_semantics == "history-conditioned-shaped"
        )


@dataclass(frozen=True)
class PolicyExpectationPlan:
    """Bound compute for paired stochastic continuations under a frozen state grid."""

    state_grid: ContinuationPlan
    independent_path_replicates: int
    fixed_prefix_terms: int
    hard_cap_tail_terms_per_path: int
    stopping_draws_per_path: int
    paired_alternative_paths: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("independent_path_replicates", self.independent_path_replicates),
            ("fixed_prefix_terms", self.fixed_prefix_terms),
            ("hard_cap_tail_terms_per_path", self.hard_cap_tail_terms_per_path),
            ("stopping_draws_per_path", self.stopping_draws_per_path),
        ):
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.independent_path_replicates < 2:
            raise ValueError(
                "policy expectation requires at least two independent continuation paths"
            )
        if self.stopping_draws_per_path < 2:
            raise ValueError("randomized truncation requires at least two stopping draws")
        if not self.paired_alternative_paths:
            raise ValueError(
                "alternatives must use paired path and stopping schedules; "
                "estimate variance from paired differences"
            )
        declared_terms = self.fixed_prefix_terms + self.hard_cap_tail_terms_per_path
        if self.state_grid.continuation_turns != declared_terms:
            raise ValueError(
                "state-grid continuation_turns must equal fixed prefix plus tail hard cap"
            )

    @property
    def bridge_step_hard_cap_per_checkpoint(self) -> int:
        grid = self.state_grid
        return len(grid.world_seeds) * len(grid.regime_ids) * (
            max(grid.decision_steps)
            + len(grid.decision_steps)
            * len(grid.alternative_ids)
            * self.independent_path_replicates
            * (self.fixed_prefix_terms + self.hard_cap_tail_terms_per_path)
        )

    def summary(self) -> dict[str, int | str | bool]:
        return {
            "independentUnit": "world",
            "independentWorldCount": self.state_grid.independent_unit_count,
            "regimeCountPerWorld": len(self.state_grid.regime_ids),
            "frozenStateCount": self.state_grid.frozen_state_count,
            "alternativeCountPerState": len(self.state_grid.alternative_ids),
            "independentPathReplicatesPerAlternative": self.independent_path_replicates,
            "fixedPrefixTermsPerPath": self.fixed_prefix_terms,
            "stoppingDrawsPerPath": self.stopping_draws_per_path,
            "pairedAlternativePaths": self.paired_alternative_paths,
            "hardCapTailTermsPerPath": self.hard_cap_tail_terms_per_path,
            "bridgeStepHardCapPerCheckpoint": self.bridge_step_hard_cap_per_checkpoint,
        }


@dataclass(frozen=True)
class ContinuationPlan:
    """A pre-result state grid and its deterministic bridge-step upper bound."""

    world_seeds: tuple[int, ...]
    regime_ids: tuple[str, ...]
    decision_steps: tuple[int, ...]
    alternative_ids: tuple[str, ...]
    continuation_turns: int

    def __post_init__(self) -> None:
        checked = tuple(validate_seeds(self.world_seeds, "Gate 3/4 continuation plan"))
        if not checked or len(checked) != len(set(checked)):
            raise ValueError("world_seeds must be nonempty and unique")
        if (
            not self.regime_ids
            or any(not regime for regime in self.regime_ids)
            or len(set(self.regime_ids)) != len(self.regime_ids)
        ):
            raise ValueError("regime_ids must be nonempty and unique")
        if (
            not self.decision_steps
            or any(
                isinstance(step, bool) or int(step) != step or step < 0
                for step in self.decision_steps
            )
            or tuple(sorted(self.decision_steps)) != self.decision_steps
            or len(set(self.decision_steps)) != len(self.decision_steps)
        ):
            raise ValueError("decision_steps must be unique sorted nonnegative integers")
        if (
            not self.alternative_ids
            or any(not alternative for alternative in self.alternative_ids)
            or len(set(self.alternative_ids)) != len(self.alternative_ids)
        ):
            raise ValueError("alternative_ids must be nonempty and unique")
        if self.continuation_turns <= 0:
            raise ValueError("continuation_turns must be positive")

    @property
    def independent_unit_count(self) -> int:
        return len(self.world_seeds)

    @property
    def frozen_state_count(self) -> int:
        return len(self.world_seeds) * len(self.regime_ids) * len(self.decision_steps)

    @property
    def branch_count(self) -> int:
        return self.frozen_state_count * len(self.alternative_ids)

    @property
    def bridge_step_upper_bound(self) -> int:
        # One main path reaches every state; each fork alternative then consumes its
        # complete declared continuation.  Objectives reuse those physical branches.
        return len(self.world_seeds) * len(self.regime_ids) * (
            max(self.decision_steps)
            + len(self.decision_steps) * len(self.alternative_ids) * self.continuation_turns
        )

    def summary(self) -> dict[str, int]:
        return {
            "independentWorldCount": self.independent_unit_count,
            "regimeCountPerWorld": len(self.regime_ids),
            "frozenStateCount": self.frozen_state_count,
            "branchCount": self.branch_count,
            "continuationTurnsPerBranch": self.continuation_turns,
            "bridgeStepUpperBoundPerCheckpoint": self.bridge_step_upper_bound,
        }
