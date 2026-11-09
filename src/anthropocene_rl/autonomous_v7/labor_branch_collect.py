"""WP-B: paired actor branches recorded during PPO collection (``labor_training collect
--branch-select ...``), so the stage-2 A(s, a) head can later be fitted online on the actor's
own states (WP-C; not here).

At a selected fraction of the rollout's decision states (one camp at one world decision),
the pre-decision world is forked and two branches are run for ``k`` decisions each:

* ``a``: the joint action the rollout actually executed (the focal camp's initial actor
  sample and, if the native rejected it, the rollout's own retry sample; every other camp
  exactly as in the rollout). Branch ``a``'s first successor must reproduce the rollout's
  own successor; this is checked on every pair (``aFirstMatchesRollout``).
* ``b``: identical, except that the focal camp's initial action is the actor's SECOND,
  independent sample at the same state (keyed generator; a rejected ``b`` gets a keyed
  actor retry).

After the first decision every camp, including the focal camp, follows the continuation
policy: by default the actor itself (``continuation="actor"``, amendment B4: on-policy
targets), each camp sampled with a per-(decision, camp) keyed generator; or H12
(``continuation="h12"``, the old search semantics, via the shadow resolve). The mode is
recorded in the file and in every record.

Common random numbers
---------------------
* Engine: both branches (and every stream) restore the SAME pre-decision fork, so the
  native world, including its keyed simulation randomness, is identical at the root.
* Decision 0: the two branches submit the same joint action except the focal camp's.
  Recorded retries are reused for every camp whose initial attempt the rollout saw
  rejected; any other retry is a keyed actor sample (same key in both branches).
* Decisions >= 1: camp ``c`` at decision ``j`` of stream ``s`` samples with
  ``torch.Generator`` seeded from ``(CONTINUATION_TAG, seed, world, root turn, s, j, c)``
  and retries with ``RETRY_TAG``: independent of the branch, the focal camp and the action,
  so two identical states draw identical actions and differing states share the same
  noise. Hence branch ``a`` (the same joint action for every focal camp of a decision) is
  run ONCE per stream and shared by all of that decision's selected camps.
* No branch draw touches the rollout's sampling generator; branching never changes the
  rollout itself (the rollout's post-decision state is forked before the branches run,
  restored after, and its physical digest is checked against the rollout's successor).

Selection (D6)
--------------
``select="uniform"``: every camp-state with probability ``rate``. ``select="adaptive"``: an
H12 shadow action per state (one heuristic resolve per world decision, all camps at once;
``labor_search.NativeSearch.shadow_h12``), per-head distances between the actor's executed
action and H12's (``head_distances``), thresholds calibrated so the EXPECTED rate including
the floor is ``rate`` (``labor_a_stage2.calibrate_thresholds``: fixed ``thresholds`` from a
file, else online on the distances of all EARLIER decisions of this collection; until
``calibration_min_states`` states have been seen the rate is uniform ``rate``), probability
1 when far on some head else ``floor`` (``branch_probabilities``), a keyed Bernoulli draw
(``select_branches``, key (world, turn, camp)). Each record carries its probability and
weight 1/p; every visited state carries its probability and distances (``states``), so the
next iteration can calibrate on this one.

Targets (D3)
------------
Per branch and stream, ``folded`` = (1 - lam) T_1 + lam T_k (``labor_a_stage2.
folded_lambda_return``) bootstrapped with the rollout's own critics (the collect
checkpoint's V: "the current online V") at s_1 and s_k; ``exact`` is the exact truncated
lambda-return (rows exist at every decision; identical to the fold for k <= 2). ``diff`` =
mean over streams of G_a - G_b, per objective (qol_years, population_years, feasibility,
settlement_guard), raw units. Feasibility per decision is initial plus retry cost, as in the
rollout's GAE (``labor_training._decision_rewards``).

The warm-up actor update of amendment B is NOT implemented here: see
:func:`warmup_actor_update` (a documented stub, default off).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from . import contracts
from . import labor_a_stage2 as s2
from . import labor_sa_advantage as sa

BRANCH_FILE_KIND = "model9-labor-paired-branches-v1"
BRANCH_SCHEMA_VERSION = 1
SELECT_MODES = ("off", "uniform", "adaptive")
CONTINUATIONS = ("actor", "h12")
COMPONENTS = ("qol_years", "population_years", "feasibility", "settlement_guard")
# Policy heads (labor_ppo.LABOR_POLICY_HEADS + NONLABOR_POLICY_HEADS): also the D6 distance
# heads and the per-head action-difference flags.
HEADS = (
    "specialty",
    "activity",
    "movement",
    "fission",
    "trade_value",
    "trade_gate",
    "trade_share",
)
SECOND_SAMPLE_TAG = 0x5753  # "WS": the actor's second sample at a branched state
CONTINUATION_TAG = 0x5743  # "WC": actor continuation draws at decisions >= 1
RETRY_TAG = 0x5752  # "WR": keyed actor retry samples in branches
WARMUP_KL_CAP = 0.005  # amendment B warm-up (not implemented; see warmup_actor_update)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchConfig:
    select: str = "off"
    rate: float = s2.TARGET_BRANCH_RATE  # uniform rate; adaptive: expected rate incl. floor
    floor: float = s2.FLOOR_BRANCH_RATE
    far_probability: float = 1.0
    horizon: int = s2.MAX_PAIRED_HORIZON
    continuation: str = "actor"
    shadow: bool | None = None  # None: on iff needed (adaptive or h12 continuation)
    heads: tuple[str, ...] = HEADS
    value_scale: float = 1.0  # trade value distance unit (encoded log value)
    share_scale: float = 1.0
    thresholds: Mapping[str, float] | None = None  # fixed D6 thresholds (else online)
    calibration_min_states: int = 32
    streams: int = 1
    seed: int = 0
    store_rows: bool = True
    lam: float = s2.LAMBDA
    discount_years: float = contracts.HORIZON_YEARS
    verify_restore: bool = True
    # Physical digest of every branch successor (tests/audits; the first successor of
    # branch a is always digested, to check it against the rollout's own successor).
    record_digests: bool = False

    def __post_init__(self) -> None:
        if self.select not in SELECT_MODES:
            raise ValueError(f"branch select must be one of {SELECT_MODES}")
        if self.continuation not in CONTINUATIONS:
            raise ValueError(f"branch continuation must be one of {CONTINUATIONS}")
        if not 1 <= int(self.horizon) <= s2.MAX_PAIRED_HORIZON:
            raise ValueError(f"branch horizon must be in 1..{s2.MAX_PAIRED_HORIZON}")
        if not 0.0 < float(self.rate) <= 1.0:
            raise ValueError("branch rate must be in (0, 1]")
        if self.select == "adaptive":
            if not 0.0 < float(self.floor) < float(self.far_probability) <= 1.0:
                raise ValueError("adaptive branching needs 0 < floor < far probability <= 1")
            if not float(self.floor) <= float(self.rate) <= float(self.far_probability):
                raise ValueError("adaptive branch rate must lie between floor and far probability")
            if self.shadow is False:
                raise ValueError("adaptive branch selection needs the H12 shadow action")
        if self.continuation == "h12" and self.shadow is False:
            raise ValueError("an H12 continuation needs the H12 shadow resolve")
        if self.continuation == "h12" and int(self.streams) != 1:
            raise ValueError("H12 continuations are deterministic: use one stream")
        if int(self.streams) < 1:
            raise ValueError("branch streams must be positive")
        if not self.heads or any(h not in HEADS for h in self.heads):
            raise ValueError(f"branch heads must be a non-empty subset of {HEADS}")
        if len(set(self.heads)) != len(self.heads):
            raise ValueError("branch heads repeat")
        if not (self.value_scale > 0 and self.share_scale > 0):
            raise ValueError("scalar distance scales must be positive")
        if self.thresholds is not None and set(self.thresholds) != set(self.heads):
            raise ValueError("fixed thresholds must name exactly the branch heads")
        if int(self.calibration_min_states) < 1:
            raise ValueError("calibration needs at least one state")
        if not 0.0 <= float(self.lam) <= 1.0 or not self.discount_years > 0:
            raise ValueError("lambda must be in [0, 1] and discount years positive")
        if int(self.seed) < 0:
            raise ValueError("branch seed must be non-negative")

    @property
    def enabled(self) -> bool:
        return self.select != "off"

    @property
    def uses_shadow(self) -> bool:
        if self.shadow is not None:
            return bool(self.shadow)
        return self.select == "adaptive" or self.continuation == "h12"

    def state_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["heads"] = list(self.heads)
        out["thresholds"] = None if self.thresholds is None else dict(self.thresholds)
        out["shadowResolved"] = self.uses_shadow
        return out


# ---------------------------------------------------------------------------
# Keyed randomness
# ---------------------------------------------------------------------------


def keyed_generator(tag: int, *key: int) -> torch.Generator:
    """A torch generator seeded only by (tag, key): order- and batch-independent."""
    words = np.random.SeedSequence([int(tag), *[int(k) % (1 << 64) for k in key]])
    hi, lo = (int(x) for x in words.generate_state(2, np.uint32))
    return torch.Generator().manual_seed(((hi << 32) | lo) & ((1 << 63) - 1))


# ---------------------------------------------------------------------------
# Head distances and differences (payload level)
# ---------------------------------------------------------------------------


def _log_value(payload: Mapping[str, Any]) -> np.ndarray:
    low, high = contracts.TRADE_VALUE_RANGE
    v = np.clip(np.asarray(payload["tradeValue"], dtype=np.float64), low, high)
    return np.log(v) / max(abs(math.log(low)), abs(math.log(high)))


def head_distances(
    actor: Mapping[str, Any],
    h12: Mapping[str, Any],
    catalog: Any,
    *,
    heads: Sequence[str] = HEADS,
    value_scale: float = 1.0,
    share_scale: float = 1.0,
) -> dict[str, float]:
    """Per-head distance between two schema-12 payloads at one state (``labor_a_stage2.
    head_distance`` kinds): specialty = fraction of residents reassigned after native
    apportionment; activity = mean total variation over role rows staffed by either;
    movement/fission = 0/1; trade gate = fraction of groups whose gate differs; trade
    value = mean |encoded log value| difference / ``value_scale``; trade share = |share
    difference| / ``share_scale``."""
    from .labor_search import labor_state

    counts_a, activity_a = labor_state(actor, catalog)
    counts_b, activity_b = labor_state(h12, catalog)
    out: dict[str, float] = {}
    for head in heads:
        if head == "specialty":
            d = s2.head_distance("counts", counts_a[None], counts_b[None])
        elif head == "activity":
            staffed = (counts_a[1:] > 0) | (counts_b[1:] > 0)
            if not staffed.any():
                out[head] = 0.0
                continue
            d = s2.head_distance(
                "simplex", activity_a[staffed][None], activity_b[staffed][None]
            )
        elif head == "movement":
            d = s2.head_distance("categorical", [int(actor["movement"])], [int(h12["movement"])])
        elif head == "fission":
            d = s2.head_distance("categorical", [int(actor["fission"])], [int(h12["fission"])])
        elif head == "trade_gate":
            d = s2.head_distance("categorical", [actor["tradeGate"]], [h12["tradeGate"]])
        elif head == "trade_value":
            d = s2.head_distance(
                "scalar", _log_value(actor)[None], _log_value(h12)[None], scale=value_scale
            )
        elif head == "trade_share":
            d = s2.head_distance(
                "scalar", [float(actor["tradeShare"])], [float(h12["tradeShare"])],
                scale=share_scale,
            )
        else:  # pragma: no cover - guarded by BranchConfig
            raise ValueError(f"unknown head {head!r}")
        out[head] = float(d[0])
    return out


def head_differences(a: Mapping[str, Any], b: Mapping[str, Any], catalog: Any) -> dict[str, bool]:
    """Which policy heads differ between two executed payloads (distance > 0 exactly)."""
    return {h: d > 0.0 for h, d in head_distances(a, b, catalog).items()}


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


@dataclass
class _Selection:
    probability: np.ndarray
    calibrated: bool
    thresholds: dict[str, float] | None
    far: np.ndarray | None


class BranchSelector:
    """D6 selection with probabilities fixed before the state's own draw."""

    def __init__(self, config: BranchConfig) -> None:
        self.config = config
        self.history: dict[str, list[float]] = {h: [] for h in config.heads}
        self.last_calibration: s2.BranchCalibration | None = None

    def probabilities(self, distances: Sequence[Mapping[str, float]] | None) -> _Selection:
        c = self.config
        n = 0 if distances is None else len(distances)
        if c.select == "uniform":
            return _Selection(np.full(n, float(c.rate)), False, None, None)
        assert distances is not None, "adaptive selection needs distances"
        thresholds = None if c.thresholds is None else dict(c.thresholds)
        calibrated = thresholds is not None
        if thresholds is None and len(self.history[c.heads[0]]) >= int(c.calibration_min_states):
            self.last_calibration = s2.calibrate_thresholds(
                {h: np.asarray(v) for h, v in self.history.items()},
                target_rate=float(c.rate),
                floor_rate=float(c.floor),
                far_probability=float(c.far_probability),
            )
            thresholds = dict(self.last_calibration.thresholds)
            calibrated = True
        if not calibrated or n == 0:
            return _Selection(np.full(n, float(c.rate)), False, thresholds, None)
        current = {h: np.asarray([d[h] for d in distances]) for h in c.heads}
        far = s2.far_mask(current, thresholds)
        p = s2.branch_probabilities(
            current, thresholds, floor_rate=float(c.floor), far_probability=float(c.far_probability)
        )
        return _Selection(np.asarray(p, dtype=np.float64), True, thresholds, far)

    def observe(self, distances: Sequence[Mapping[str, float]]) -> None:
        """Add a decision's distances AFTER its selection (no self-dependence)."""
        for d in distances:
            for h in self.config.heads:
                self.history[h].append(float(d[h]))


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


@dataclass
class BranchTelemetry:
    worlds: int = 0
    decisions: int = 0  # rollout world decisions seen by the hook (= rollout native steps)
    camp_states: int = 0
    branched_states: int = 0
    shadow_steps: int = 0
    branch_steps: int = 0
    branch_runs: int = 0  # world branches run (one shared a per stream + one b per focal)
    forks: int = 0
    restores: int = 0
    world_seconds: float = 0.0
    shadow_seconds: float = 0.0
    selection_seconds: float = 0.0
    branch_seconds: float = 0.0
    restore_seconds: float = 0.0

    def state_dict(self) -> dict[str, Any]:
        overhead = (
            self.shadow_seconds + self.selection_seconds + self.branch_seconds
            + self.restore_seconds
        )
        per = self.branched_states or 0
        return {
            "worlds": self.worlds,
            "decisions": self.decisions,
            "campStates": self.camp_states,
            "branchedStates": self.branched_states,
            "nativeSteps": {
                "rollout": self.decisions,
                "shadow": self.shadow_steps,
                "branch": self.branch_steps,
            },
            "branchRuns": self.branch_runs,
            "forks": self.forks,
            "restores": self.restores,
            "seconds": {
                "worlds": self.world_seconds,
                "shadow": self.shadow_seconds,
                "selection": self.selection_seconds,
                "branch": self.branch_seconds,
                "forkRestoreVerify": self.restore_seconds,
                "rolloutDerived": max(0.0, self.world_seconds - overhead),
            },
            "branchSecondsPerBranchedState": self.branch_seconds / per if per else None,
            "branchStepsPerBranchedState": self.branch_steps / per if per else None,
            "overheadSecondsPerBranchedState": overhead / per if per else None,
        }


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


def default_shadow_factory(bridge: Any, world: int, options: Any) -> tuple[Callable, Callable]:
    """The native H12 shadow: a heuristic ``Model9Bridge`` reset from the labor bridge's
    captured snapshot and stepped with its own decisions (``NativeSearch.shadow_h12``):
    one heuristic resolve per call, all live camps at once. Returns (shadow, close)."""
    from .bridge import Model9Bridge
    from .labor_search import NativeSearch, SearchConfig

    heuristic = Model9Bridge(economic_diagnostics=True)
    native = NativeSearch(bridge, heuristic, world=world, options=options, config=SearchConfig())
    return native.shadow_h12, heuristic.close


# Patched by the mock end-to-end driver (no native heuristic there).
SHADOW_FACTORY: Callable[[Any, int, Any], tuple[Callable, Callable]] = default_shadow_factory


@dataclass
class _Plan:
    turn: int
    step: Mapping[str, Any]
    live: list[Mapping[str, Any]]
    catalogs: dict[int, Any]
    rows: dict[int, int]
    tensors: Mapping[str, torch.Tensor]
    raw: Mapping[str, torch.Tensor] | None
    output: Mapping[str, torch.Tensor]
    retained: Mapping[int, Any]
    founded: dict[int, int]
    selected: list[tuple[int, float, dict[str, Any]]]  # (camp, probability, selection info)
    pre_fork: int
    labels: Mapping[int, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class PairedBranchCollector:
    """Hook object for ``labor_training.collect_labor_rollout(brancher=...)``."""

    def __init__(
        self,
        config: BranchConfig,
        *,
        actor: Any,
        task_critic: Any,
        cost_critics: Any,
        normalizer: Any,
        reset_options: Any,
        action_binding: Mapping[str, Any] | None = None,
        shadow_factory: Callable[[Any, int, Any], tuple[Callable, Callable]] | None = None,
    ) -> None:
        if not config.enabled:
            raise ValueError("a branch collector needs --branch-select uniform or adaptive")
        self.config = config
        self.actor = actor
        self.task = task_critic
        self.costs = cost_critics
        self.normalizer = normalizer
        self.reset_options = reset_options
        self.action_binding = action_binding
        self.shadow_factory = shadow_factory
        self.selector = BranchSelector(config)
        self.telemetry = BranchTelemetry()
        self.records: list[dict[str, Any]] = []
        self.states: list[dict[str, Any]] = []
        self.bridge: Any = None
        self.world = -1
        self.world_index = -1
        self.personality: Any = None
        self._shadow: Callable | None = None
        self._close: Callable | None = None
        self._world_started = 0.0

    # -- world lifecycle ------------------------------------------------------
    def begin_world(self, bridge: Any, *, seed: int, world_index: int, personality: Any) -> None:
        self.bridge = bridge
        self.world = int(seed)
        self.world_index = int(world_index)
        self.personality = personality
        self.telemetry.worlds += 1
        self._world_started = time.perf_counter()
        if self.config.uses_shadow:
            factory = self.shadow_factory or SHADOW_FACTORY
            self._shadow, self._close = factory(bridge, self.world, self.reset_options)

    def end_world(self) -> None:
        self.telemetry.world_seconds += time.perf_counter() - self._world_started
        if self._close is not None:
            self._close()
        self._shadow = self._close = None
        self.bridge = None

    # -- native primitives (counted) -----------------------------------------
    def _fork(self) -> int:
        fork, _ = self.bridge.fork()
        self.telemetry.forks += 1
        return int(fork)

    def _restore(self, fork: int) -> Mapping[str, Any]:
        self.telemetry.restores += 1
        return self.bridge.restore(fork)

    def _shadow_labels(self, step: Mapping[str, Any], catalogs: Mapping[int, Any]) -> dict:
        started = time.perf_counter()
        labels = self._shadow(step, catalogs)
        self.telemetry.shadow_steps += 1
        self.telemetry.shadow_seconds += time.perf_counter() - started
        return labels

    def _execute(self, catalogs, retained, retry_factory) -> Any:
        from . import labor_training as lt

        execution = lt.execute_labor_decision(self.bridge, dict(catalogs), retained, retry_factory)
        self.telemetry.branch_steps += 1
        return execution

    # -- model helpers ----------------------------------------------------------
    def _view(self, step: Mapping[str, Any], only: set[int] | None = None) -> dict | None:
        """Live villages, catalogs and actor/critic tensors of every live camp (or of the
        camps in ``only``: rows are built independently, so their rows are the same)."""
        from . import labor_training as lt

        live = [
            v for v in step["villages"]
            if v.get("observation") is not None and not v.get("terminated")
            and (only is None or int(v["camp"]) in only)
        ]
        if not live:
            return None
        contexts, tensors = lt.labor_policy_tensors(
            live, self.normalizer, personality=self.personality
        )
        raw = lt._rollout_privileged(live, self.task, self.costs)
        catalogs = {ctx[0].camp_id: ctx[0] for ctx in contexts}
        rows = {int(v["camp"]): i for i, v in enumerate(live)}
        return {"live": live, "catalogs": catalogs, "tensors": tensors, "raw": raw, "rows": rows}

    def _values(self, tensors, raw, row: int) -> list[float]:
        from . import labor_training as lt

        t = lt._one_row(tensors, row)
        r = None if raw is None else lt._one_row(raw, row)
        task = lt._task_values(self.task, t, r)
        costs = lt._cost_values(self.costs, t, r)
        values = {k: float(v[0]) for k, v in (*task.items(), *costs.items())}
        return [values[name] for name in COMPONENTS]

    def _sample(self, output_row, tensors_row, generator: torch.Generator) -> dict[str, Any]:
        """One full actor sample (labor + non-labor) for one row, scored per head."""
        from .labor_model import (
            sample_labor_policy,
            sample_nonlabor_policy,
            schema12_initial_action,
        )

        labor = sample_labor_policy(output_row, tensors_row, generator=generator)
        nonlabor = sample_nonlabor_policy(output_row, generator=generator)
        return self._scored(
            output_row, tensors_row, labor, nonlabor, schema12_initial_action(labor, nonlabor, 0)
        )

    def _scored(self, output_row, tensors_row, labor, nonlabor, payload) -> dict[str, Any]:
        from .labor_model import labor_log_prob, nonlabor_log_prob

        lab = labor_log_prob(output_row, labor, tensors_row)
        non = nonlabor_log_prob(output_row, nonlabor)
        head_logp = {h: float(lab[h][0]) for h in ("specialty", "activity")}
        head_logp.update({h: float(non[h][0]) for h in HEADS if h not in head_logp})
        return {
            "payload": dict(payload),
            "labor": {k: v.detach().cpu().clone() for k, v in labor.items()},
            "nonlabor": {k: v.detach().cpu().clone() for k, v in nonlabor.items()},
            "laborLogp": float(lab["joint"][0]),
            "nonlaborLogp": float(non["joint"][0]),
            "headLogp": head_logp,
        }

    def _retained(self, sample: Mapping[str, Any]) -> Any:
        from .labor_ppo import RetainedLaborAction

        return RetainedLaborAction(
            sample["payload"], sample["labor"], sample["laborLogp"],
            sample["nonlabor"], sample["nonlaborLogp"],
        )

    def _keyed_retry(self, village: Mapping[str, Any], catalog: Any, attempt: Any, key) -> Any:
        """The rollout's retry factory, with a keyed generator instead of the rollout's."""
        from . import labor_training as lt
        from .labor_model import labor_log_prob, sample_labor_policy, schema12_retry_action
        from .labor_ppo import RetainedLaborAction

        contexts, tensors = lt.labor_policy_tensors(
            [village], self.normalizer, personality=self.personality, rejected_attempts=[attempt]
        )
        if contexts[0][0].digest != catalog.digest:
            raise ValueError("branch retry tensor context differs from typed rejection")
        output = self.actor(tensors)
        sample = sample_labor_policy(output, tensors, generator=keyed_generator(RETRY_TAG, *key))
        logp = float(labor_log_prob(output, sample, tensors)["joint"].detach().cpu()[0])
        return RetainedLaborAction(
            schema12_retry_action(sample, 0), lt._one_row(sample, 0), logp, None, None
        )

    # -- rollout hook -------------------------------------------------------------
    def before_decision(
        self,
        *,
        step: Mapping[str, Any],
        live: Sequence[Mapping[str, Any]],
        contexts: Sequence[Any],
        tensors: Mapping[str, torch.Tensor],
        raw: Mapping[str, torch.Tensor] | None,
        output: Mapping[str, torch.Tensor],
        retained: Mapping[int, Any],
        founded: Mapping[int, int],
    ) -> _Plan | None:
        """Select camps at this rollout decision; fork the pre-decision world if any.
        Reads only; never touches the rollout's generator."""
        started = time.perf_counter()
        c = self.config
        turn = int(step["turn"])
        catalogs = {ctx[0].camp_id: ctx[0] for ctx in contexts}
        rows = {int(v["camp"]): i for i, v in enumerate(live)}
        camps = [int(v["camp"]) for v in live]
        self.telemetry.decisions += 1
        self.telemetry.camp_states += len(camps)
        labels = None
        distances = None
        shadow_before = self.telemetry.shadow_seconds
        if self._shadow is not None:
            labels = self._shadow_labels(step, catalogs)
            distances = [
                head_distances(
                    retained[camp].payload, labels[camp].action, catalogs[camp],
                    heads=c.heads, value_scale=c.value_scale, share_scale=c.share_scale,
                )
                for camp in camps
            ]
        if c.select == "adaptive":
            selection = self.selector.probabilities(distances)
        else:
            selection = self.selector.probabilities([{}] * len(camps))
        keys = [(self.world, turn, camp) for camp in camps]
        chosen, weights = s2.select_branches(selection.probability, keys, int(c.seed))
        if distances is not None:
            self.selector.observe(distances)
        selected = []
        for i, camp in enumerate(camps):
            info = {
                "mode": c.select,
                "probability": float(selection.probability[i]),
                "weight": float(weights[i]),
                "calibrated": bool(selection.calibrated),
                "far": None if selection.far is None else bool(selection.far[i]),
                "distances": None if distances is None else dict(distances[i]),
                # Heads over their threshold at THIS decision's calibration.
                "triggers": (
                    None if distances is None or selection.thresholds is None
                    else [h for h in c.heads if distances[i][h] > selection.thresholds[h]]
                ),
            }
            self.states.append(
                {"world": self.world, "turn": turn, "camp": camp, "selected": bool(chosen[i]),
                 **info}
            )
            if chosen[i]:
                info["thresholds"] = selection.thresholds
                selected.append((camp, float(selection.probability[i]), info))
        self.telemetry.selection_seconds += (
            time.perf_counter() - started - (self.telemetry.shadow_seconds - shadow_before)
        )
        if not selected:
            return None
        clock = time.perf_counter()
        pre = self._fork()
        self.telemetry.restore_seconds += time.perf_counter() - clock
        self.telemetry.branched_states += len(selected)
        return _Plan(
            turn=turn, step=step, live=list(live), catalogs=catalogs, rows=rows,
            tensors=tensors, raw=raw, output=output, retained=dict(retained),
            founded=dict(founded), selected=selected, pre_fork=pre, labels=labels,
        )

    def after_decision(self, plan: _Plan, execution: Any) -> None:
        """Run the selected pairs from the pre-decision fork, then return the bridge to the
        rollout's successor (checked) and drop both forks.

        Branch ``a`` is the same joint action for every focal camp of this decision and
        continuation keys do not name the focal camp, so ONE ``a`` run per stream serves
        all of them (each focal camp reads its own rewards/values from it); each focal
        camp gets its own ``b`` run. Cost per decision: (m + 1) k S world steps for m
        selected camps, not 2 m k S."""
        from .labor_search import physical_digest

        clock = time.perf_counter()
        post = self._fork()
        self.telemetry.restore_seconds += time.perf_counter() - clock
        rollout_digest = physical_digest(execution.step)
        started = time.perf_counter()
        try:
            self.records.extend(self._pairs(plan, execution, rollout_digest))
        finally:
            self.telemetry.branch_seconds += time.perf_counter() - started
            clock = time.perf_counter()
            back = self._restore(post)
            if self.config.verify_restore and physical_digest(back) != rollout_digest:
                raise RuntimeError(
                    "branching changed the rollout: the restored post-decision world differs "
                    "from the rollout's successor"
                )
            self.bridge.drop(post)
            self.bridge.drop(plan.pre_fork)
            self.telemetry.restore_seconds += time.perf_counter() - clock

    # -- the pairs of one decision ------------------------------------------------------
    @torch.no_grad()
    def _pairs(self, plan: _Plan, execution: Any, rollout_digest: str) -> list[dict]:
        from . import labor_training as lt

        c = self.config
        focals = [camp for camp, _p, _info in plan.selected]
        rollout_retries = dict(getattr(execution, "retry", {}) or {})
        villages = {int(v["camp"]): v for v in plan.live}

        def retry0(stream: int, keyed: set[int]):
            def factory(camp, catalog, attempt):
                if camp in rollout_retries and camp not in keyed:
                    return rollout_retries[camp]
                key = (c.seed, self.world, plan.turn, stream, 0, camp)
                return self._keyed_retry(villages[camp], catalog, attempt, key)

            return factory

        clock = time.perf_counter()
        steps = self.telemetry.branch_steps
        shared_a = [
            self._branch(plan, focals, stream, dict(plan.retained), retry0(stream, set()),
                         digest_first=(stream == 0))
            for stream in range(int(c.streams))
        ]
        a_cost = {"nativeSteps": self.telemetry.branch_steps - steps,
                  "seconds": time.perf_counter() - clock, "sharedBy": len(focals)}
        a_first = shared_a[0]["digest0"]
        records = []
        for focal, _probability, info in plan.selected:
            clock = time.perf_counter()
            steps = self.telemetry.branch_steps
            row = plan.rows[focal]
            catalog = plan.catalogs[focal]
            output_row = lt._one_row(plan.output, row)
            tensors_row = lt._one_row(plan.tensors, row)
            a_retained = plan.retained[focal]
            # The executed payload itself (never re-serialized: a one-row float32 sigmoid or
            # exp may differ from the batch kernel by an ulp), scored under the behaviour
            # policy.
            a = self._scored(
                output_row, tensors_row, a_retained.labor_sample, a_retained.nonlabor_sample,
                a_retained.payload,
            )
            b = self._second_sample(plan, focal, output_row, tensors_row, a)
            b_runs = []
            for stream in range(int(c.streams)):
                first = dict(plan.retained)
                first[focal] = self._retained(b)
                b_runs.append(self._branch(plan, [focal], stream, first, retry0(stream, {focal})))
            branches = {
                "a": [run["focals"][focal] for run in shared_a],
                "b": [run["focals"][focal] for run in b_runs],
            }
            targets = branch_targets(branches, int(c.horizon), lam=float(c.lam),
                                     discount_years=float(c.discount_years))
            records.append({
                "world": self.world,
                "worldIndex": self.world_index,
                "turn": plan.turn,
                "camp": focal,
                "population": int(catalog.population),
                "continuation": c.continuation,
                "k": int(c.horizon),
                "streams": int(c.streams),
                "selection": info,
                "actions": {
                    "a": {**a, "retried": focal in rollout_retries,
                          "features": self._features(a["payload"], catalog)},
                    "b": {**b, "features": self._features(b["payload"], catalog)},
                },
                "headDiffers": head_differences(a["payload"], b["payload"], catalog),
                "identicalActions": a["payload"] == b["payload"],
                "root": {
                    "value": self._values(plan.tensors, plan.raw, row),
                    "tensors": tensors_row if c.store_rows else None,
                    "privileged": (
                        None if plan.raw is None or not c.store_rows
                        else lt._one_row(plan.raw, row)
                    ),
                },
                "branches": branches,
                "targets": targets,
                "aFirstMatchesRollout": a_first == rollout_digest,
                "work": {
                    "b": {"nativeSteps": self.telemetry.branch_steps - steps,
                          "seconds": time.perf_counter() - clock},
                    "aShared": dict(a_cost),
                },
            })
        return records

    def _second_sample(self, plan: _Plan, focal: int, output_row, tensors_row, a) -> dict:
        """The actor's second, independent sample at the branched state (keyed)."""
        return self._sample(
            output_row, tensors_row,
            keyed_generator(SECOND_SAMPLE_TAG, self.config.seed, self.world, plan.turn, focal),
        )

    def _features(self, payload: Mapping[str, Any], catalog: Any) -> torch.Tensor | None:
        if self.action_binding is None:
            return None
        from .labor_action_adapter import canonical_action_features

        return canonical_action_features(payload, catalog, **self.action_binding).clone()

    def _focal_entry(self, focal, decision, decision_turn, execution, next_step, founded,
                     rewards, terminal, years, view) -> dict[str, Any]:
        from . import labor_training as lt

        outcome = {int(v["camp"]): v for v in next_step["villages"]}
        guard = lt._settlement_guard(outcome[focal], founded[focal], int(next_step["turn"]))
        retried = focal in (getattr(execution, "retry", {}) or {})
        feasibility = float(execution.initial_attempts[focal].feasibility_cost) + (
            float(execution.final_attempts[focal].feasibility_cost) if retried else 0.0
        )
        entry: dict[str, Any] = {
            "decision": decision,
            "turn": decision_turn,
            "rewards": [
                float(rewards[focal]["qol_years"]),
                float(rewards[focal]["population_years"]),
                feasibility,
                float(guard),
            ],
            "years": float(years[focal]),
            "focalRetried": bool(retried),
            "digest": None,
            "value": None,
            "row": None,
        }
        if terminal[focal]:
            entry["status"] = "terminated"
        elif view is not None and focal in view["rows"]:
            entry["status"] = "live"
            r = view["rows"][focal]
            entry["value"] = self._values(view["tensors"], view["raw"], r)
            if self.config.store_rows:
                entry["row"] = {
                    "tensors": lt._one_row(view["tensors"], r),
                    "privileged": None if view["raw"] is None else lt._one_row(view["raw"], r),
                }
        else:
            entry["status"] = "inactive"
        return entry

    def _branch(self, plan: _Plan, focals: Sequence[int], stream: int,
                first: Mapping[int, Any], first_retry: Callable, *,
                digest_first: bool = False) -> dict[str, Any]:
        """One world branch on one stream: decision 0 = ``first``, then the continuation,
        for up to ``horizon`` decisions while some focal camp is live. Returns, per focal
        camp, its branch record (its decisions end at its own termination/inactivity)."""
        from . import labor_training as lt
        from .labor_bc import _idle_retry
        from .labor_search import _retained as retain_payload
        from .labor_search import physical_digest

        c = self.config
        self._restore(plan.pre_fork)
        self.telemetry.branch_runs += 1
        founded = dict(plan.founded)
        catalogs = dict(plan.catalogs)
        retained, retry_factory = dict(first), first_retry
        decision_turn = plan.turn
        out = {f: {"stream": stream, "decisions": [], "terminated": False} for f in focals}
        open_focals = list(focals)
        digest0 = None
        for decision in range(int(c.horizon)):
            execution = self._execute(catalogs, retained, retry_factory)
            next_step = execution.step
            rewards, terminal, years = lt._transition_rewards(
                next_step, tuple(open_focals), founded, world_index=self.world_index,
                decision_turn=decision_turn,
            )
            digest = (
                physical_digest(next_step)
                if c.record_digests or (digest_first and decision == 0) else None
            )
            if decision == 0:
                digest0 = digest
            # Every live camp's catalog/tensors when another decision follows (its payloads
            # need them); only the open focal camps' rows for V at s_k.
            last = decision + 1 >= int(c.horizon)
            view = self._view(next_step, set(open_focals) if last else None)
            still_open = []
            for focal in open_focals:
                entry = self._focal_entry(focal, decision, decision_turn, execution, next_step,
                                          founded, rewards, terminal, years, view)
                entry["digest"] = digest
                out[focal]["decisions"].append(entry)
                if entry["status"] == "terminated":
                    out[focal]["terminated"] = True
                if entry["status"] == "live":
                    still_open.append(focal)
            open_focals = still_open
            if not open_focals or last:
                break
            decision_turn = int(next_step["turn"])
            catalogs = dict(view["catalogs"])
            if c.continuation == "actor":
                retained, retry_factory = self._actor_actions(
                    view, plan.turn, stream, decision + 1
                )
            else:
                labels = self._shadow_labels(next_step, catalogs)
                retained = {
                    camp: retain_payload(labels[camp].action, catalogs[camp])
                    for camp in sorted(catalogs)
                }
                retry_factory = _idle_retry
        return {"focals": out, "digest0": digest0}

    def _actor_actions(self, view, root_turn: int, stream: int, decision: int):
        """Every live camp samples from the actor with its own keyed generator (B4). The key
        names neither the branch nor the focal camp: identical states draw identically."""
        from . import labor_training as lt

        c = self.config
        output = self.actor(view["tensors"])
        retained = {}
        for camp, row in view["rows"].items():
            if camp not in view["catalogs"]:
                continue
            g = keyed_generator(CONTINUATION_TAG, c.seed, self.world, root_turn, stream,
                                decision, camp)
            sample = self._sample(lt._one_row(output, row), lt._one_row(view["tensors"], row), g)
            retained[camp] = self._retained(sample)
        villages = {int(v["camp"]): v for v in view["live"]}

        def retry(camp, catalog, attempt):
            key = (c.seed, self.world, root_turn, stream, decision, camp)
            return self._keyed_retry(villages[camp], catalog, attempt, key)

        return retained, retry

    # -- output ---------------------------------------------------------------------
    def payload(self, provenance: Mapping[str, Any]) -> dict[str, Any]:
        calibration = self.selector.last_calibration
        return {
            "kind": BRANCH_FILE_KIND,
            "schemaVersion": BRANCH_SCHEMA_VERSION,
            "provenance": dict(provenance),
            "config": self.config.state_dict(),
            "components": list(COMPONENTS),
            "heads": list(HEADS),
            "rng": {
                "selection": {"tag": s2.BRANCH_TAG, "key": "[tag, seed, world, turn, camp]"},
                "secondSample": {"tag": SECOND_SAMPLE_TAG, "key": "[tag, seed, world, turn, camp]"},
                "continuation": {
                    "tag": CONTINUATION_TAG,
                    "key": "[tag, seed, world, root turn, stream, decision, camp]",
                },
                "retry": {
                    "tag": RETRY_TAG,
                    "key": "[tag, seed, world, root turn, stream, decision, camp]",
                },
                "engine": "both branches restore the same pre-decision native fork",
                "sharedA": "one branch-a run per (decision, stream) serves every focal camp",
            },
            "calibration": None if calibration is None else {
                **{k: v for k, v in asdict(calibration).items()},
            },
            "telemetry": self.telemetry.state_dict(),
            "states": list(self.states),
            "records": list(self.records),
        }


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def succ_stream(branch: Mapping[str, Any], offset: int) -> tuple[sa.SuccStream, list[list[float]]]:
    """A recorded branch as ``labor_sa_advantage.SuccStream`` plus its successor V rows
    (global row = ``offset`` + local index)."""
    decisions = branch["decisions"]
    values: list[list[float]] = []
    row: dict[int, int] = {}
    status: dict[int, str] = {}
    for entry in decisions:
        j = int(entry["decision"]) + 1
        status[j] = str(entry["status"])
        if entry["value"] is not None:
            row[j] = offset + len(values)
            values.append([float(x) for x in entry["value"]])
    stream = sa.SuccStream(
        stream=int(branch["stream"]),
        rewards=np.asarray([e["rewards"] for e in decisions], dtype=np.float64).reshape(-1, 4),
        years=np.asarray([e["years"] for e in decisions], dtype=np.float64),
        terminated=bool(branch["terminated"]),
        status=status,
        row=row,
    )
    return stream, values


def branch_targets(
    branches: Mapping[str, Sequence[Mapping[str, Any]]],
    k: int,
    *,
    lam: float = s2.LAMBDA,
    discount_years: float = contracts.HORIZON_YEARS,
) -> dict[str, Any]:
    """Folded (D3, primary) and exact lambda-returns per branch, and their CRN difference.

    ``folded``/``exact`` are None when any stream's target is unavailable (an inactive
    focal camp or a missing successor row)."""
    streams: dict[str, list[sa.SuccStream]] = {}
    rows: list[list[float]] = []
    for name in ("a", "b"):
        streams[name] = []
        for branch in branches[name]:
            stream, values = succ_stream(branch, len(rows))
            rows.extend(values)
            streams[name].append(stream)
    table = np.asarray(rows, dtype=np.float64).reshape(-1, 4)
    v_succ = {h: table[:, i] for i, h in enumerate(COMPONENTS)}
    out: dict[str, Any] = {"k": int(k), "lam": float(lam), "discountYears": float(discount_years),
                           "primary": "folded"}
    for method in ("folded", "exact"):
        per: dict[str, list[np.ndarray]] = {"a": [], "b": []}
        ok = True
        for name in ("a", "b"):
            for stream in streams[name]:
                if method == "folded":
                    g = s2.folded_lambda_return(
                        stream, k, v_succ, lam=lam, discount_years=discount_years
                    )
                else:
                    g = s2.exact_lambda_return(
                        stream, k, v_succ, lam=lam, discount_years=discount_years
                    )
                    if g is s2._MISSING_ROW:
                        g = None
                if g is None:
                    ok = False
                    break
                per[name].append(np.asarray(g, dtype=np.float64))
            if not ok:
                break
        if not ok:
            out[method] = None
            continue
        ga, gb = np.mean(per["a"], axis=0), np.mean(per["b"], axis=0)
        out[method] = {
            "a": [float(x) for x in ga],
            "b": [float(x) for x in gb],
            "diff": [
                float(x)
                for x in np.mean(
                    [x - y for x, y in zip(per["a"], per["b"], strict=True)], axis=0
                )
            ],
        }
    return out


# ---------------------------------------------------------------------------
# Amendment B warm-up (stub)
# ---------------------------------------------------------------------------


def warmup_actor_update(*_: Any, **__: Any) -> None:
    """NOT IMPLEMENTED (default off; nothing calls it).

    Amendment B's warm-up: small actor updates from paired-branch differences before GAE
    takes over. Per branched state the two sampled actions get the leave-one-out
    advantages ``+diff`` (a) and ``-diff`` (b) per objective
    (``labor_a_stage2.paired_policy_advantages``), combined with the rollout row's
    personality weights and cost multipliers (``labor_ppo.combine_labor_advantages``),
    row weight 1/p (``record["selection"]["weight"]``), and the per-head surrogate over the
    behaviour log-probs recorded here (``record["actions"][x]["headLogp"]``) with a KL cap
    of ``WARMUP_KL_CAP`` (0.005) per update. It needs a second scoring path in
    ``labor_training train`` (the b action is not a rollout parent), so it belongs with
    WP-C's train-side reader rather than in collection.
    """
    raise NotImplementedError(
        "the amendment-B warm-up actor update is not implemented (WP-B records its inputs; "
        "see labor_branch_collect.warmup_actor_update)"
    )


def branch_path_for(rollout: Any) -> Any:
    """``<rollout stem>.branches.pt`` next to the rollout."""
    from pathlib import Path

    path = Path(rollout)
    if path.suffix == ".pt":
        return path.with_name(path.stem + ".branches.pt")
    return path.with_name(path.name + ".branches.pt")
