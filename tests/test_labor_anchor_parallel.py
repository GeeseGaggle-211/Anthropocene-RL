"""labor_anchor_producer --jobs: (world, path) units in separate processes, same payload.

Synthetic CPU fixtures: a deterministic fake bridge, real labor models, a scripted
continuation; the pool is replaced by an in-process serial executor (the real one spawns
processes that need the native bridge). No sealed or reserved worlds.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import subprocess
import sys
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest
import torch
from test_labor_a_ppo_wiring import _default_models
from test_labor_ppo_perhead import _village_at

from anthropocene_rl.autonomous_v7 import labor_anchor_producer as producer
from anthropocene_rl.autonomous_v7.gate34_continuation_collector import freeze_geometric_schedule
from anthropocene_rl.autonomous_v7.labor_bc import LaborResetOptions
from anthropocene_rl.autonomous_v7.labor_model import LaborPersonality

REPO = Path(__file__).resolve().parents[1]
PREPILOT = "68ae43d"
WORLDS = [50000, 50001]
RESETS: list[int] = []
ORIGINAL_RUN = producer._run_continuation  # before any fixture patches it


class Bridge:
    """Deterministic per world; ``drift`` makes every reset differ (non-reproducible)."""

    drift = False

    def reset(self, world, **kw):
        RESETS.append(world)
        tag = f"w{world}" + (f"-{len(RESETS)}" if self.drift else "")
        self.step = {"turn": 0, "fingerprint": tag,
                     "villages": [_village_at(7)]}
        return self.step

    def runtime_identity(self):
        return {"fixture": True}

    def fork(self):
        return "fork", self.step

    def restore(self, fork):
        return self.step

    def drop(self, fork):
        pass

    def close(self):
        pass


class DriftingBridge(Bridge):
    drift = True


class SerialExecutor:
    def __init__(self, workers, threads):
        self.workers, self.threads, self.calls = workers, threads, 0

    def submit(self, fn, *args):
        self.calls += 1
        future = Future()
        try:
            future.set_result(fn(*args))
        except BaseException as error:  # noqa: BLE001 - mirrored into the future
            future.set_exception(error)
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        pass


def _install(module, monkeypatch, models):
    layout, actor, task, costs, normalizer = models

    def continuation(**kw):
        seed = kw["initial_seed"]
        step = kw["step"]
        successor = module._successor_inputs(step, normalizer, kw["personality"])
        privileged = module._privileged_inputs(step, task, costs)
        camps = kw["opening_camps"]
        return module.LaborContinuationPath(
            kw["world_seed"], kw["path_index"], kw["alternative"], kw["schedule"],
            {camp: successor[camp] for camp in camps},
            {camp: f"{seed}/{camp}" for camp in camps},
            {camp: {head: (float(seed % 97) + camp, float(kw["world_seed"] % 13) - camp)
                    for head in module.LABOR_VALUE_HEADS} for camp in camps},
            {camp: (20.0, 20.0) for camp in camps},
            {camp: (False, True) for camp in camps},
            {camp: privileged[camp] for camp in camps},
        )

    monkeypatch.setattr(module, "_models_from_checkpoint", lambda state: models)
    monkeypatch.setattr(module, "labor_checkpoint_personality", lambda state: LaborPersonality())
    monkeypatch.setattr(module, "_run_continuation", continuation)


@pytest.fixture()
def models(monkeypatch, tmp_path):
    actor, task, costs, normalizer, layout = _default_models()
    models = (layout, actor, task, costs, normalizer)
    _install(producer, monkeypatch, models)
    source = tmp_path / "source.pt"
    torch.save({}, source)
    RESETS.clear()
    return {"models": models, "source": source}


def _collect(module, source, bridge=Bridge, **kw):
    return module.collect_labor_anchor_arrays(
        checkpoint_path=source, state={}, world_seeds=WORLDS, optimizer_seed=1,
        continuation_seed_base=880000, role="train-anchor", path_count=3,
        bridge_factory=bridge, reset_options=LaborResetOptions(players=16, width=128,
                                                               height=64, separation=9),
        **kw,
    )


def _equal(left, right):
    assert set(left) == set(right)
    for key in left:
        assert left[key].dtype == right[key].dtype, key
        assert np.array_equal(left[key], right[key]), key


def _prepilot_producer():
    modname = "anthropocene_rl.autonomous_v7._prepilot_labor_anchor_producer"
    if modname in sys.modules:
        return sys.modules[modname]
    try:
        source = subprocess.run(
            ["git", "show",
             f"{PREPILOT}:src/anthropocene_rl/autonomous_v7/labor_anchor_producer.py"],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("pre-pilot source is not available from git")
    spec = importlib.util.spec_from_loader(modname, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "anthropocene_rl.autonomous_v7"
    sys.modules[modname] = module
    exec(compile(source, f"<{PREPILOT}:labor_anchor_producer.py>", "exec"), module.__dict__)
    return module


def test_sequential_payload_is_bitwise_the_prepilot_producer(models, monkeypatch):
    old = _prepilot_producer()
    _install(old, monkeypatch, models["models"])
    _equal(_collect(old, models["source"]), _collect(producer, models["source"]))
    assert RESETS == WORLDS * 2


@pytest.mark.parametrize("jobs", [2, 6, 64])
def test_parallel_units_reproduce_the_sequential_payload(models, jobs):
    sequential = _collect(producer, models["source"])
    created = []

    def factory(workers, threads):
        created.append(SerialExecutor(workers, threads))
        return created[-1]

    RESETS.clear()
    parallel = _collect(producer, models["source"], jobs=jobs, threads=3,
                        executor_factory=factory)
    _equal(sequential, parallel)
    (executor,) = created
    assert executor.workers == min(jobs, 6) and executor.threads == 3 and executor.calls == 6
    assert RESETS == [50000] * 3 + [50001] * 3  # one fresh reset per (world, path) unit
    meta = json.loads(str(parallel["metadata_json"]))
    assert meta["source"]["nativeRuntimeIdentities"] == {"50000": {"fixture": True},
                                                         "50001": {"fixture": True}}


def test_parallel_refuses_a_non_reproducible_reset(models):
    _collect(producer, models["source"], bridge=DriftingBridge)  # sequential: one reset/world
    with pytest.raises(ValueError, match="different reset decisions"):
        _collect(producer, models["source"], bridge=DriftingBridge, jobs=2,
                 executor_factory=SerialExecutor)


def test_worker_refuses_another_checkpoint_or_policy(models, tmp_path):
    def swap(workers, threads):
        models["source"].write_bytes(b"changed")
        return SerialExecutor(workers, threads)

    with pytest.raises(ValueError, match="checkpoint changed"):
        _collect(producer, models["source"], jobs=2, executor_factory=swap)


def test_jobs_validation_and_cli_defaults(models):
    for jobs in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            _collect(producer, models["source"], jobs=jobs)
    args = producer.argument_parser().parse_args(
        ["--checkpoint", "c", "--worlds", "1", "--optimizer-seed", "1",
         "--continuation-seed-base", "5000", "--role", "train-anchor", "--out", "o"]
    )
    assert args.jobs == 1 and args.threads == 4 and args.progress is False


def test_unit_job_and_results_pickle(models):
    schedule = freeze_geometric_schedule(schedule_id="x", rng_seed=880000, draw_count=16,
                                         survival_probability=0.95, hard_cap_terms=200)
    job = {"bridgeFactory": producer.LaborModel9Bridge, "schedule": schedule,
           "reset": LaborResetOptions(players=16, width=128, height=64, separation=9)}
    back = pickle.loads(pickle.dumps(job))
    assert back["schedule"] == schedule and back["bridgeFactory"] is producer.LaborModel9Bridge
    assert back["reset"].state_dict() == job["reset"].state_dict()
    _, actor, task, costs, normalizer = models["models"]
    parent = producer._world_parent(Bridge(), 50000, job["reset"], normalizer,
                                     LaborPersonality(), task, costs)
    again = pickle.loads(pickle.dumps(parent))
    assert again.parent_digest == parent.parent_digest and again.opening == parent.opening
    assert pickle.loads(pickle.dumps(producer._anchor_unit)) is producer._anchor_unit


def test_progress_lines_are_stderr_only(models, capsys):
    quiet = _collect(producer, models["source"])
    loud = _collect(producer, models["source"], progress=True)
    _equal(quiet, loud)
    lines = [json.loads(line) for line in capsys.readouterr().err.splitlines()
             if line.startswith('{"anchorProgress"')]
    assert len(lines) == 2 * 3 * 2
    assert {line["anchorProgress"]["alternative"] for line in lines} == {
        "sampled-policy", "paired-sample"}


def test_term_progress_is_opt_in_and_the_plan_line_lists_every_path(models, capsys):
    import inspect

    assert inspect.signature(ORIGINAL_RUN).parameters["progress_every"].default == 0
    seen = []
    original = producer._run_continuation

    def spy(**kw):
        seen.append(kw.get("progress_every"))
        return original(**kw)

    producer._run_continuation = spy
    try:
        _collect(producer, models["source"])
        _collect(producer, models["source"], progress=True)
    finally:
        producer._run_continuation = original
    assert seen == [None] * 12 + [producer.PROGRESS_EVERY_TERMS] * 12
    (plan,) = [json.loads(line)["anchorPlan"] for line in capsys.readouterr().err.splitlines()
               if line.startswith('{"anchorPlan"')]
    assert plan["role"] == "train-anchor" and plan["paths"] == 3
    assert sorted(plan["maximumTerms"]) == [f"{w}/{p}" for w in WORLDS for p in range(3)]
