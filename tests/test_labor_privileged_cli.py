"""Intervention A (learner): privileged normalizer fit, initialization, migration, calibration.

The live test is the bounded local end-to-end path: teacher shards with the
privileged view on one fitting and one calibration world, the frozen per-family
fit with its dead-mask report, ``initialize --campaign-r5 --privileged-normalizer``,
one pretrain-critic epoch and critic-calibration. The migration test converts a
checkpoint without a privileged block through the strict-load migration hook.
"""

from __future__ import annotations

import json

import pytest
import torch

from anthropocene_rl.autonomous_v7 import labor_teacher_data as ltd
from anthropocene_rl.autonomous_v7.bridge import DEFAULT_BRIDGE
from anthropocene_rl.autonomous_v7.labor_model import BOUNDED_FEATURE_CONTRACT
from anthropocene_rl.autonomous_v7.labor_privileged import (
    LABOR_PRIVILEGED_LAYOUT_V1,
    PRIVILEGED_NORMALIZER_KIND,
)

pytestmark = pytest.mark.skipif(not DEFAULT_BRIDGE.is_file(), reason="model9 bridge is not built")


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    """Bounded local data: 1 fitting + 1 calibration world, 8 players, 2 decisions, zero tail."""
    from test_labor_bc import _fit_test_normalizer

    from anthropocene_rl.autonomous_v7.bridge import Model9Bridge

    with Model9Bridge() as bridge:
        if bridge.privileged_layout != LABOR_PRIVILEGED_LAYOUT_V1:
            pytest.skip("bridge does not emit privileged schema 1")
    root = tmp_path_factory.mktemp("privileged-cli")
    norm = root / "norm.pt"
    torch.save(
        {
            "feature_contract": BOUNDED_FEATURE_CONTRACT,
            "normalizer": _fit_test_normalizer().state_dict(),
        },
        norm,
    )
    for world, name in ((90100, "fit"), (90180, "cal")):
        assert (
            ltd.main(
                [
                    "--normalizer-checkpoint",
                    str(norm),
                    "--worlds",
                    str(world),
                    "--turns",
                    "2",
                    "--players",
                    "8",
                    "--jobs",
                    "1",
                    "--allow-dirty",
                    "--return-tail",
                    "truncated-zero",
                    "--out",
                    str(root / name),
                ]
            )
            == 0
        )
    return root, norm


def test_live_privileged_fit_initialize_pretrain_and_calibrate(live, capsys) -> None:
    from anthropocene_rl.autonomous_v7.labor_pretrain_eval import main as eval_main
    from anthropocene_rl.autonomous_v7.labor_training import main as training_main

    root, norm = live
    with pytest.raises(ValueError, match="calibration world"):
        training_main(
            [
                "fit-privileged-normalizer",
                "--dataset",
                str(root / "cal"),
                "--out",
                str(root / "bad.pt"),
            ]
        )
    pn = root / "pn.pt"
    training_main(
        [
            "fit-privileged-normalizer",
            "--dataset",
            str(root / "fit"),
            "--calibration-dataset",
            str(root / "cal"),
            "--report",
            str(root / "dead.json"),
            "--out",
            str(pn),
        ]
    )
    fitted = torch.load(pn, weights_only=True)
    assert fitted["kind"] == PRIVILEGED_NORMALIZER_KIND
    assert fitted["privileged_normalizer"]["fit"]["worlds"] == [90100]
    report = json.loads((root / "dead.json").read_text())
    assert set(report["data_audit"]["families"]) == {
        f.name for f in LABOR_PRIVILEGED_LAYOUT_V1.families
    }

    init = root / "init.pt"
    training_main(
        [
            "initialize",
            "--normalizer-checkpoint",
            str(norm),
            "--privileged-normalizer",
            str(pn),
            "--campaign-r5",
            "--world-seed",
            "90100",
            "--optimizer-seed",
            "1",
            "--out",
            str(init),
        ]
    )
    state = torch.load(init, weights_only=True)
    assert state["labor_model_contract"]["privilegedCritic"]["schemaVersion"] == 1
    assert state["labor_model_contract"]["criticValueSupport"]["high"] == 2.5
    assert not any("privileged" in key for key in state["actor"])

    critic = root / "critic.pt"
    training_main(
        [
            "pretrain-critic",
            "--checkpoint",
            str(init),
            "--dataset",
            str(root / "fit"),
            "--epochs",
            "1",
            "--batch-size",
            "256",
            "--lr",
            "3e-4",
            "--out",
            str(critic),
        ]
    )
    summary = torch.load(critic, weights_only=True)["heuristic_critic_pretraining"]
    assert summary["privileged_schema_version"] == 1 and summary["epochs"] == 1
    trained = torch.load(critic, weights_only=True)
    assert all(torch.equal(trained["actor"][k], state["actor"][k]) for k in state["actor"])

    out = root / "calibration.json"
    assert (
        eval_main(
            [
                "critic-calibration",
                "--checkpoint",
                str(critic),
                "--dataset",
                str(root / "cal"),
                "--fit-dataset",
                str(root / "fit"),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    calibration = json.loads(out.read_text())
    assert calibration["privileged_critic"]["normalizer_fit_worlds"] == [90100]
    assert set(calibration["verdict"]) == {
        "qol_years",
        "population_years",
        "feasibility",
        "settlement_guard",
    }
    # A calibration dataset that fitted the privileged moments is not held out.
    with pytest.raises(ValueError, match="not held out"):
        eval_main(
            [
                "critic-calibration",
                "--checkpoint",
                str(critic),
                "--dataset",
                str(root / "fit"),
                "--out",
                str(root / "leak.json"),
            ]
        )


def test_live_convert_privileged_migrates_through_the_strict_load_hook(live) -> None:
    from anthropocene_rl.autonomous_v7.labor_critic_support import main as support_main
    from anthropocene_rl.autonomous_v7.labor_training import _models_from_checkpoint
    from anthropocene_rl.autonomous_v7.labor_training import main as training_main

    root, norm = live
    pn = root / "pn-migrate.pt"
    training_main(["fit-privileged-normalizer", "--dataset", str(root / "fit"), "--out", str(pn)])
    plain = root / "plain.pt"
    training_main(
        [
            "initialize",
            "--normalizer-checkpoint",
            str(norm),
            "--campaign-r5",
            "--world-seed",
            "90100",
            "--optimizer-seed",
            "2",
            "--out",
            str(plain),
        ]
    )
    converted = root / "converted.pt"
    assert (
        support_main(
            [
                "convert-privileged",
                "--checkpoint",
                str(plain),
                "--privileged-normalizer",
                str(pn),
                "--seed",
                "7",
                "--out",
                str(converted),
            ]
        )
        == 0
    )
    before = torch.load(plain, weights_only=True)
    after = torch.load(converted, weights_only=True)
    record = after["critic_privileged_conversions"][-1]
    assert record["privileged_schema_version"] == 1 and record["init_seed"] == 7
    for group in ("actor", "task_critic", "cost_critics"):
        for key, value in before[group].items():
            assert torch.equal(after[group][key], value), (group, key)
        added = set(after[group]) - set(before[group])
        assert all("privileged_" in key for key in added)
        assert bool(added) == (group != "actor")
    layout, _, task, costs, _ = _models_from_checkpoint(after)
    assert layout.privileged == LABOR_PRIVILEGED_LAYOUT_V1
    assert task.privileged_normalizer is costs.privileged_normalizer
    with pytest.raises(ValueError, match="already carries"):
        support_main(
            [
                "convert-privileged",
                "--checkpoint",
                str(converted),
                "--privileged-normalizer",
                str(pn),
                "--out",
                str(root / "twice.pt"),
            ]
        )
