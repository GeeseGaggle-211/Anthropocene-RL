"""Labor FiLM conditioning and declared personality sampling (docs/LABOR_PERSONALITY.md)."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from test_autonomous_v7_round3_search import observation as legacy_observation
from test_labor_bridge import _catalog_payload
from test_labor_model import _labor_tensors, _layout
from test_labor_training import _row

from anthropocene_rl.autonomous_v7.bridge import observation_batch
from anthropocene_rl.autonomous_v7.labor_bridge import (
    LABOR_ACTIVITY_FEATURE_WIDTH,
    LABOR_RESOURCE_FEATURE_WIDTH,
    LABOR_SPECIALTY_FEATURE_WIDTH,
)
from anthropocene_rl.autonomous_v7.labor_model import (
    LaborActor,
    LaborCostCritics,
    LaborEncoderLayout,
    LaborModelContract,
    LaborObservationEncoder,
    LaborPersonality,
    LaborTaskCritic,
    labor_checkpoint_payload,
    labor_checkpoint_personality,
    labor_film_extent,
    labor_personality_rows,
    labor_policy_tensors,
    load_labor_checkpoint_models,
    set_labor_film_bound,
)
from anthropocene_rl.autonomous_v7.labor_personality import (
    LaborPersonalityAssignment,
    LaborPersonalityConditioning,
    parse_personality_distribution,
    personality_for_world,
)
from anthropocene_rl.autonomous_v7.labor_ppo import combine_labor_advantages
from anthropocene_rl.autonomous_v7.labor_training import build_labor_training_batch
from anthropocene_rl.autonomous_v7.model import Normalizer

PRE_FILM_COMMIT = "87b99d4"
REPOSITORY = Path(__file__).resolve().parents[1]


def _models(seed: int = 5):
    torch.manual_seed(seed)
    layout = _layout()
    return (
        LaborActor(layout, width=24, token_width=16),
        LaborTaskCritic(layout, width=24, token_width=16),
        LaborCostCritics(layout, width=20, token_width=16),
    )


def _outputs(actor, task, costs, tensors):
    with torch.no_grad():
        result = {f"actor.{k}": v for k, v in actor(tensors).items()}
        result.update({f"task.{k}": v for k, v in task(tensors).items()})
        result.update({f"cost.{k}": v for k, v in costs(tensors).items()})
    return result


def _assert_bitwise(left, right) -> None:
    assert set(left) == set(right)
    for name in left:
        assert torch.equal(left[name], right[name]), name


def _encoders(*modules):
    return [
        m for module in modules for m in module.modules() if isinstance(m, LaborObservationEncoder)
    ]


def _randomize_film(*modules, scale: float = 0.5, seed: int = 11) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for encoder in _encoders(*modules):
            for film in encoder._films():
                film.weight.copy_(torch.randn(film.weight.shape, generator=generator) * scale)


class _PreFilm:
    """Temporarily route every labor encoder through its exact pre-FiLM path."""

    def __init__(self, *modules):
        self.encoders = _encoders(*modules)

    def __enter__(self):
        for encoder in self.encoders:
            encoder.forward = lambda tensors, e=encoder: e.encode(tensors, film=False)
        return self

    def __exit__(self, *exc):
        for encoder in self.encoders:
            del encoder.forward


@pytest.mark.parametrize("train_mode", [False, True])
def test_zero_condition_is_bitwise_the_prefilm_computation(train_mode: bool) -> None:
    actor, task, costs = _models()
    for module in (actor, task, costs):
        module.train(train_mode)
    tensors = _labor_tensors()
    with _PreFilm(actor, task, costs):
        reference = _outputs(actor, task, costs, tensors)
    _assert_bitwise(_outputs(actor, task, costs, tensors), reference)
    # Even trained (random, nonzero) FiLM weights are the identity at c = 0.
    _randomize_film(actor, task, costs)
    _assert_bitwise(_outputs(actor, task, costs, tensors), reference)


def _prefilm_module():
    try:
        source = subprocess.run(
            ["git", "show", f"{PRE_FILM_COMMIT}:src/anthropocene_rl/autonomous_v7/labor_model.py"],
            cwd=REPOSITORY,
            capture_output=True,
            check=True,
            text=True,
            encoding="utf-8",
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("pre-FiLM labor_model source is not available in this checkout")
    import tempfile

    directory = Path(tempfile.mkdtemp())
    path = directory / "labor_model_prefilm.py"
    path.write_text(source, encoding="utf-8")
    name = "anthropocene_rl.autonomous_v7._labor_model_prefilm"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_new_model_matches_the_committed_prefilm_model_bitwise() -> None:
    old = _prefilm_module()
    torch.manual_seed(5)
    layout = old.LaborEncoderLayout(*_layout().state_dict().values())
    old_models = (
        old.LaborActor(layout, width=24, token_width=16),
        old.LaborTaskCritic(layout, width=24, token_width=16),
        old.LaborCostCritics(layout, width=20, token_width=16),
    )
    new_models = _models(5)
    # Same seed, same historical parameters: FiLM creation does not consume the global RNG.
    for old_module, new_module in zip(old_models, new_models, strict=True):
        new_state = new_module.state_dict()
        for name, value in old_module.state_dict().items():
            assert torch.equal(new_state[name], value), name
        assert {n for n in new_state if n not in old_module.state_dict()} == {
            n for n in new_state if ".film_" in n
        }
    tensors = _labor_tensors()
    _assert_bitwise(_outputs(*new_models, tensors), _outputs(*old_models, tensors))


def test_prefilm_checkpoint_loads_with_identity_film() -> None:
    actor, task, costs = _models()
    tensors = _labor_tensors()
    expected = _outputs(actor, task, costs, tensors)
    old_payload = {
        key: {name: value for name, value in module.state_dict().items() if ".film_" not in name}
        for key, module in (("actor", actor), ("task_critic", task), ("cost_critics", costs))
    }
    fresh = _models(99)
    _randomize_film(*fresh)
    fresh[0].load_state_dict(old_payload["actor"], strict=True)
    fresh[1].load_state_dict(old_payload["task_critic"], strict=True)
    fresh[2].load_state_dict(old_payload["cost_critics"], strict=True)
    for encoder in _encoders(*fresh):
        assert all(bool((film.weight == 0).all()) for film in encoder._films())
    _assert_bitwise(_outputs(*fresh, tensors), expected)


def _real_villages():
    raw = legacy_observation(seed=92)
    raw["core"][40:56] = [0.0] * 16
    raw["core"][144] = 0.0
    raw.update(laborCatalog=_catalog_payload(), laborPrevious=None)
    return [{"camp": 5, "society": 11, "parentSociety": 0, "observation": raw}]


def test_strict_checkpoint_loader_accepts_prefilm_tensors_and_declared_distribution() -> None:
    villages = _real_villages()
    normalizer = Normalizer.fit_bounded(observation_batch(villages))
    conditioning = LaborPersonalityConditioning("dirichlet:1.0", key=4, film_bound=3.0)
    assignment = conditioning.assignment(0)
    _, tensors = labor_policy_tensors(
        villages, normalizer, personality=personality_for_world(assignment, 50000)
    )
    pair = assignment.world(50000).table[11]
    assert torch.equal(tensors["labor_personality"], torch.tensor([pair], dtype=torch.float32))
    layout = LaborEncoderLayout(
        tensors["core"].shape[1],
        LABOR_SPECIALTY_FEATURE_WIDTH,
        LABOR_ACTIVITY_FEATURE_WIDTH,
        LABOR_RESOURCE_FEATURE_WIDTH,
    )
    torch.manual_seed(1)
    models = (
        LaborActor(layout, width=16, token_width=16),
        LaborTaskCritic(layout, width=16, token_width=16),
        LaborCostCritics(layout, width=16, token_width=16),
    )
    set_labor_film_bound(3.0, *models)
    state = labor_checkpoint_payload(*models, normalizer, layout, personality=assignment)
    assert state["labor_model_contract"]["personalityConditioning"] == conditioning.state_dict()
    for key in ("actor", "task_critic", "cost_critics"):
        state[key] = {n: v for n, v in state[key].items() if ".film_" not in n}
    torch.manual_seed(2)
    fresh = (
        LaborActor(layout, width=16, token_width=16),
        LaborTaskCritic(layout, width=16, token_width=16),
        LaborCostCritics(layout, width=16, token_width=16),
    )
    contract, _ = load_labor_checkpoint_models(state, *fresh, expected_layout=layout)
    assert contract.conditioning == conditioning
    assert all(e.film_bound == 3.0 for e in _encoders(*fresh))
    _assert_bitwise(_outputs(*fresh, tensors), _outputs(*models, tensors))


def test_random_film_weights_change_outputs_for_nonzero_condition() -> None:
    actor, task, costs = _models()
    tensors = _labor_tensors()
    tensors["labor_personality"] = torch.tensor([[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]])
    before = _outputs(actor, task, costs, tensors)
    for group in ("film_local", "film_global", "film_trunk"):
        generator = torch.Generator().manual_seed(17)
        with torch.no_grad():
            for encoder in _encoders(actor, task, costs):
                for film in getattr(encoder, group):
                    film.weight.copy_(torch.randn(film.weight.shape, generator=generator))
        after = _outputs(actor, task, costs, tensors)
        heads = ["actor.fission", "task.qol_years", "cost.feasibility"]
        if group != "film_trunk":  # token outputs bypass the trunk
            heads.append("actor.specialty_logits")
        for prefix in heads:
            assert not torch.allclose(after[prefix][:2], before[prefix][:2]), (group, prefix)
            # The balanced row (c = 0) is untouched by any FiLM weights.
            assert torch.equal(after[prefix][2], before[prefix][2]), (group, prefix)
        with torch.no_grad():
            for encoder in _encoders(actor, task, costs):
                for film in getattr(encoder, group):
                    film.weight.zero_()


def test_film_bound_limits_gamma_and_beta() -> None:
    actor, task, costs = _models()
    _randomize_film(actor, task, costs, scale=50.0)
    tensors = _labor_tensors()
    tensors["labor_personality"] = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    assert labor_film_extent(actor, tensors) > 1.0
    set_labor_film_bound(0.5, actor, task, costs)
    assert labor_film_extent(actor, tensors) <= 0.5
    assert all(e.film_bound == 0.5 for e in _encoders(actor, task, costs))
    with pytest.raises(ValueError):
        set_labor_film_bound(0.0, actor)


def _villages(*rows):
    return [{"camp": 100 + s, "society": s, "parentSociety": p} for s, p in rows]


def test_dirichlet_founders_are_keyed_and_order_independent() -> None:
    conditioning = LaborPersonalityConditioning("dirichlet:1.0", key=7)
    villages = _villages((1, 0), (2, 0), (3, 3))
    first = conditioning.assignment(0).world(50000).resolve(villages)
    again = conditioning.assignment(0).world(50000).resolve(list(reversed(villages)))
    assert first == list(reversed(again))
    assert len(set(first)) == 3
    assert all(abs(q + p - 1.0) < 1e-12 and 0 <= q <= 1 for q, p in first)
    other_key = LaborPersonalityConditioning("dirichlet:1.0", key=8).assignment(0)
    assert other_key.world(50000).resolve(villages) != first
    assert conditioning.assignment(1).world(50000).resolve(villages) != first
    assert conditioning.assignment(0).world(50001).resolve(villages) != first


def test_daughters_inherit_with_small_keyed_drift() -> None:
    conditioning = LaborPersonalityConditioning("dirichlet:1.0", key=3)
    world = conditioning.assignment(0).world(50000)
    (parent,) = world.resolve(_villages((4, 0)))
    (daughter,) = world.resolve(_villages((9, 4)))
    assert daughter != parent
    assert abs(daughter[0] - parent[0]) < 0.15  # concentration 200: spread ~0.035
    # Deterministic per key, and independent of whether the parent arrived first.
    together = conditioning.assignment(0).world(50000).resolve(_villages((9, 4), (4, 0)))
    assert together == [daughter, parent]
    copies = LaborPersonalityConditioning("dirichlet:1.0", key=3, daughter_drift=0.0)
    copy_world = copies.assignment(0).world(50000)
    assert copy_world.resolve(_villages((4, 0), (9, 4))) == [parent, parent]
    # Grand-daughters chain through their parent.
    (grand,) = world.resolve(_villages((12, 9)))
    assert abs(grand[0] - daughter[0]) < 0.15


def test_balanced_is_fixed_and_reads_no_lineage() -> None:
    world = LaborPersonalityConditioning().assignment(0).world(50000)
    assert world.resolve([{}, {}]) == [(0.5, 0.5), (0.5, 0.5)]
    rows = labor_personality_rows([{}, {}], world)
    assert torch.equal(rows, labor_personality_rows([{}, {}], LaborPersonality()))
    assert personality_for_world(LaborPersonality(), 50000) == LaborPersonality()


def test_distribution_parsing_and_conditioning_round_trip() -> None:
    assert parse_personality_distribution("balanced") == ("balanced", None)
    assert parse_personality_distribution("dirichlet:2") == ("dirichlet:2.0", 2.0)
    for bad in ("fixed:0.5,0.5", "dirichlet:0", "dirichlet:x", "dirichlet"):
        with pytest.raises(ValueError):
            parse_personality_distribution(bad)
    conditioning = LaborPersonalityConditioning("dirichlet:1", key=5, film_bound=2.0)
    assert LaborPersonalityConditioning.from_state(conditioning.state_dict()) == conditioning
    # Default contracts keep their pre-FiLM bytes; declared ones round-trip.
    assert "personalityConditioning" not in LaborModelContract().state_dict()
    declared = LaborModelContract(conditioning=conditioning)
    assert LaborModelContract.from_state(declared.state_dict()) == declared


def test_checkpoint_personality_resolves_fixed_or_assignment() -> None:
    old = {"labor_model_contract": {"personality": LaborPersonality().state_dict()}}
    assert labor_checkpoint_personality(old) == LaborPersonality()
    conditioning = LaborPersonalityConditioning("dirichlet:1", key=5)
    new = {
        "iteration": 3,
        "labor_model_contract": LaborModelContract(conditioning=conditioning).state_dict(),
    }
    assignment = labor_checkpoint_personality(new)
    assert isinstance(assignment, LaborPersonalityAssignment)
    assert assignment.iteration == 3 and assignment.conditioning == conditioning


def test_per_row_weights_reproduce_fixed_balanced_bitwise() -> None:
    torch.manual_seed(0)
    task = {"qol_years": torch.randn(64) * 30, "population_years": torch.randn(64) * 9e4}
    costs = {"feasibility": torch.randn(64), "settlement_guard": torch.randn(64)}
    multipliers = {"feasibility": 1.0, "settlement_guard": 0.5}
    fixed = combine_labor_advantages(
        task, costs, personality=LaborPersonality(), cost_multipliers=multipliers
    )
    rows = combine_labor_advantages(
        task,
        costs,
        personality=LaborPersonality(),
        cost_multipliers=multipliers,
        row_weights=torch.full((64, 2), 0.5),
    )
    assert torch.equal(fixed, rows)
    weights = torch.rand(64, 1)
    weights = torch.cat([weights, 1 - weights], dim=1)
    personal = combine_labor_advantages(
        task,
        costs,
        personality=LaborPersonality(),
        cost_multipliers=multipliers,
        row_weights=weights,
    )
    from anthropocene_rl.autonomous_v7.labor_model import LABOR_OBJECTIVE_REFERENCE_SCALES as S

    manual = (
        weights[:, 0].double() / S["qol_years"] * task["qol_years"].double()
        + weights[:, 1].double() / S["population_years"] * task["population_years"].double()
        - costs["feasibility"].double()
        - 0.5 * costs["settlement_guard"].double()
    )
    assert torch.allclose(personal.double(), manual, rtol=1e-5, atol=1e-6)


def test_training_batch_uses_each_rows_own_personality() -> None:
    def personal(turn: int, weights: list[float], retry: bool):
        row = _row(turn, retry=retry, terminated=False)
        row.camp_id = turn  # one decision per camp: independent sequences
        row.initial_tensors["labor_personality"] = torch.tensor([weights])
        if row.retry_tensors is not None:
            row.retry_tensors["labor_personality"] = torch.tensor([weights])
        return row

    bootstrap = {
        (50000, camp): {
            "qol_years": 0.0,
            "population_years": 0.0,
            "feasibility": 0.0,
            "settlement_guard": 0.0,
        }
        for camp in (1, 2, 3)
    }
    kwargs = dict(
        personality=LaborPersonality(),
        cost_multipliers={"feasibility": 0.0, "settlement_guard": 0.0},
        behavior_policy_sha256="a" * 64,
        critic_sha256="b" * 64,
        normalization_sha256="c" * 64,
        lam=0.0,
    )
    weights = [[1.0, 0.0], [0.25, 0.75], [0.6, 0.4]]
    retry = [False, True, True]

    def batch_for(row_weights):
        pairs = zip(row_weights, retry, strict=True)
        rows = [personal(i + 1, w, r) for i, (w, r) in enumerate(pairs)]
        return build_labor_training_batch(rows, bootstrap, **kwargs)[0]

    batch = batch_for(weights)
    qol = batch_for([[1.0, 0.0]] * 3)
    population = batch_for([[0.0, 1.0]] * 3)
    # A pure-QoL Elder's combined advantage is the scale-normalised QoL advantage alone.
    assert float(batch.initial_advantage[0]) == pytest.approx((2.0 - 10.0) / 43.152, rel=1e-5)
    w = torch.tensor(weights)
    expected_initial = w[:, 0] * qol.initial_advantage + w[:, 1] * population.initial_advantage
    assert torch.allclose(batch.initial_advantage, expected_initial, rtol=1e-5, atol=1e-9)
    parents = batch.retry_parent
    assert parents.tolist() == [1, 2]
    expected_retry = (
        w[parents, 0] * qol.retry_advantage + w[parents, 1] * population.retry_advantage
    )
    assert torch.allclose(batch.retry_advantage, expected_retry, rtol=1e-5, atol=1e-9)
    assert not torch.allclose(batch.retry_advantage, batch_for([[0.5, 0.5]] * 3).retry_advantage)
    mismatched = personal(3, [1.0, 0.0], True)
    mismatched.retry_tensors["labor_personality"] = torch.tensor([[0.0, 1.0]])
    with pytest.raises(ValueError, match="retry personality"):
        build_labor_training_batch([mismatched], {(50000, 3): bootstrap[(50000, 1)]}, **kwargs)


def test_manifest_threads_personality_into_initialize_only() -> None:
    sys.path.insert(0, str(REPOSITORY / "tools" / "vmqueue"))
    import labor_3x5_manifest as lm

    default = lm.build_arm_phases("labor-arm2-bc-baseline", 170)
    declared = lm.build_arm_phases(
        "labor-arm2-bc-baseline", 170, personality="dirichlet:1.0", film_bound=2.0
    )
    assert all("--personality" not in phase["run"] for phase in default)
    init = next(phase for phase in declared if phase["name"] == "phase0a-init")
    assert "--personality dirichlet:1.0" in init["run"] and "--film-bound 2.0" in init["run"]
    assert sum("--personality" in phase["run"] for phase in declared) == 1
    with pytest.raises(ValueError):
        lm.build_arm_phases("labor-arm2-bc-baseline", 170, personality="fixed:1,0")


def test_rollout_and_bc_collectors_thread_one_world_resolver() -> None:
    from unittest.mock import patch

    from anthropocene_rl.autonomous_v7 import labor_bc, labor_training
    from anthropocene_rl.autonomous_v7.labor_personality import WorldPersonalities

    assignment = LaborPersonalityConditioning("dirichlet:1.0", key=2).assignment(0)
    seen: list = []

    class Bridge:
        def __init__(self, *args, **kwargs):
            pass

        def reset(self, seed, **_options):
            village = {"camp": 1, "society": 1, "parentSociety": 0, "observation": {}}
            return {"turn": 0, "villages": [village]}

        def close(self):
            pass

    def step(**kwargs):
        seen.append(("step", kwargs["world_seed"], kwargs["personality"]))
        return kwargs["step"], [object()]

    def bootstrap(step_, normalizer, task, costs, personality):
        seen.append(("bootstrap", None, personality))
        return {}

    with (
        patch.object(labor_training, "_sample_world_step", side_effect=step),
        patch.object(labor_training, "_bootstrap_values", side_effect=bootstrap),
    ):
        labor_training.collect_labor_rollout(
            actor=None,
            task_critic=None,
            cost_critics=None,
            normalizer=None,
            world_seeds=[50000, 50001],
            turns=2,
            generator=torch.Generator(),
            personality=assignment,
            bridge_factory=Bridge,
        )
    resolvers = [entry[2] for entry in seen]
    assert all(isinstance(r, WorldPersonalities) for r in resolvers)
    assert resolvers[0] is resolvers[1] is resolvers[2] is assignment.world(50000)
    assert resolvers[3] is assignment.world(50001)

    captured: list = []

    def tensors(live, normalizer, *, personality):
        captured.append(personality)
        raise RuntimeError("stop after the hook")

    class Shadow(Bridge):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    with (
        patch.object(labor_bc, "labor_policy_tensors", side_effect=tensors),
        pytest.raises(RuntimeError, match="stop after the hook"),
    ):
        labor_bc.collect_heuristic_bc_dataset(
            normalizer=None,
            world_seeds=[50002],
            turns=1,
            personality=assignment,
            labor_bridge_factory=Shadow,
            heuristic_bridge_factory=Shadow,
        )
    assert captured == [assignment.world(50002)]


def test_behavior_signature_is_unchanged_for_fixed_and_extended_for_distributions() -> None:
    from anthropocene_rl.autonomous_v7.labor_training import labor_behavior_configuration

    fixed = labor_behavior_configuration(LaborPersonality())
    assert "personalityConditioning" not in fixed
    conditioning = LaborPersonalityConditioning("dirichlet:1.0", key=2)
    declared = labor_behavior_configuration(conditioning.assignment(4))
    assert declared["personalityConditioning"] == conditioning.state_dict()
    assert declared["personalityIteration"] == 4
    stripped = {
        k: v
        for k, v in declared.items()
        if k not in {"personalityConditioning", "personalityIteration"}
    }
    assert stripped == fixed
