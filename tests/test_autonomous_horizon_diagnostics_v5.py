"""Bounded technical selection must include final models and intermediate extremes."""

import importlib.util
import sys
from pathlib import Path


def load_driver():
    directory = Path("scripts").resolve()
    spec = importlib.util.spec_from_file_location(
        "horizon_diagnostics", directory / "run_autonomous_horizon_diagnostics_v5.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(directory))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(directory))
    return module


def campaign():
    cells = []
    for variant in ("small_step", "residual_layernorm"):
        for horizon in (100, 400):
            for seed in (29, 43):
                name = f"{variant}-{horizon}-{seed}"
                frames = []
                for step in (0, 2048, 4096):
                    frames.append(
                        dict(
                            transitions=step,
                            checkpoint=dict(path=f"{name}-{step}.pt", sha256="pin"),
                            health=dict(
                                actor_layers=[dict(saturated_fraction=0.9 if step == 0 else 0.1)],
                                categorical=dict(
                                    enabled=dict(
                                        eligible=32,
                                        maximum_probability=(0.9 if step == 2048 else 0.1),
                                    )
                                ),
                            ),
                        )
                    )
                cells.append(dict(id=name, variant=variant, status="completed", health=frames))
    driver = load_driver()
    return dict(cells=cells, selected_extremes=driver.health_extremes(cells))


def test_diagnostic_union_bounds_work_and_covers_extremes_before_final():
    driver = load_driver()
    jobs = driver.diagnostic_jobs(campaign())
    assert len(jobs) == 12
    assert sum(len(j["tasks"]) for j in jobs) == 66
    assert sum(len(j["tasks"]) * 64 * 32 for j in jobs) == 135168
    assert all(set(driver.CORE) <= set(j["tasks"]) for j in jobs)
    assert sum(len(j["tasks"]) == 13 for j in jobs) == 2
    assert sum(j["checkpoint"]["path"].endswith("4096.pt") for j in jobs) == 8


def test_diagnostic_union_deduplicates_checkpoint_task():
    driver = load_driver()
    data = campaign()
    for cell in data["cells"]:
        cell["health"] = cell["health"][-1:]
    data["selected_extremes"] = driver.health_extremes(data["cells"])
    jobs = driver.diagnostic_jobs(data)
    assert len(jobs) == 8
    assert sum(len(j["tasks"]) for j in jobs) == 50
