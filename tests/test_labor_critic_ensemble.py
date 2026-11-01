"""Critic-ensemble evaluation: ``critic-calibration --checkpoint a.pt b.pt [--member-reports]``.

Synthetic tiny critics (no bridge, no teacher data): the averaging rule, the identical-members
and single-path invariants, fail-closed validation and the end-to-end report.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_labor_privileged_model import _critics, _layout
from test_labor_qol_head import _training_dataset

from anthropocene_rl.autonomous_v7 import labor_pretrain_eval as pe
from anthropocene_rl.autonomous_v7.labor_critic_ensemble import (
    EnsembleMember,
    EnsembleTask,
    ensemble_critic_predictions,
    validate_ensemble,
)
from anthropocene_rl.autonomous_v7.labor_model import (
    LABOR_COSTS,
    LABOR_OBJECTIVES,
    LaborCostCritics,
    LaborQoLHead,
    LaborTaskCritic,
    LaborValueSupport,
)

REFERENCE = {"value": 10.0, "source": "artifact", "digest": "a" * 64}


class _Norm:
    """Observation-normalizer stand-in: only its state feeds the binding digest."""

    def __init__(self, tag: str = "n") -> None:
        self.tag = tag
        self.dead: dict = {}
        self.set_mean = {"trade": np.zeros(32)}
        self.set_std = {"trade": np.ones(32)}

    def state_dict(self) -> dict:
        return {"tag": self.tag}


def _models(seed: int, *, bounded: bool = True, support=None, qol_head=None, privileged=False):
    torch.manual_seed(seed)
    layout = _layout(privileged)
    if privileged:
        task, costs = _critics(layout, seed=seed)
        return task, costs
    head = (LaborQoLHead() if bounded else None) if qol_head is None else qol_head
    task = LaborTaskCritic(layout, width=16, token_width=16, qol_head=head, support=support)
    costs = LaborCostCritics(layout, width=16, token_width=16)
    for module in (task, costs):
        module.eval()
    return task, costs


def _dataset(shift: int = 0) -> dict:
    dataset = _training_dataset()
    dataset["row_world"] = dataset["row_world"] + shift
    dataset["collection"]["dataset_digest"] = f"digest-{shift}"
    return dataset


def _member(index, task, costs, *, tag="n", corrections=None, source="checkpoint", sha="0" * 64):
    return EnsembleMember(
        index=index,
        path=Path(f"m{index}.pt"),
        sha256=sha,
        state={"heuristic_critic_pretraining": {"dataset_digest": "fit", "epochs": 1}},
        task=task,
        costs=costs,
        normalizer=_Norm(tag),
        corrections=dict(corrections or {}),
        corrections_source=source,
    )


def _dump(value) -> str:
    return json.dumps(value, sort_keys=True, default=str)


@pytest.mark.parametrize("bounded", [True, False])
def test_ensemble_predictions_are_the_mean_of_member_values(bounded: bool) -> None:
    a = _models(1, bounded=bounded)
    b = _models(2, bounded=bounded)
    c = _models(3, bounded=bounded)
    members = [_member(i, *m) for i, m in enumerate((a, b, c))]
    ensemble = validate_ensemble(members)
    dataset = _dataset()
    rows = np.arange(6)
    observed = {name: 5.0 for name in LABOR_OBJECTIVES}
    each = [pe.critic_predictions(t, k, dataset, rows, 4, observed) for t, k in (a, b, c)]
    mean = ensemble_critic_predictions(ensemble.task, ensemble.costs, dataset, rows, 4, observed)
    assert set(mean) == set(each[0])
    for key in mean:
        np.testing.assert_allclose(mean[key], sum(e[key] for e in each) / 3.0, rtol=1e-12, atol=0)
    # The members really differ, so the mean is not any one of them.
    assert not np.allclose(each[0]["qol_years"], each[1]["qol_years"])
    for name in (*LABOR_OBJECTIVES, *LABOR_COSTS):
        assert not np.array_equal(mean[name], each[0][name])


def test_population_mixture_expectation_equals_the_mean_of_member_expectations() -> None:
    a, b = _models(1), _models(2)
    ensemble = validate_ensemble([_member(0, *a), _member(1, *b)])
    rows = np.arange(6)
    dataset = _dataset()
    mixture = pe._head_probabilities(ensemble.task, dataset, rows, "population_years", 4)
    assert mixture.shape == (6, a[0].support.bins)
    np.testing.assert_allclose(mixture.sum(axis=1), 1.0, rtol=1e-12)
    from anthropocene_rl.autonomous_v7.labor_model import (
        LABOR_OBJECTIVE_REFERENCE_SCALES,
        symexp,
    )

    values = symexp(a[0].bins.double()).numpy()
    scale = LABOR_OBJECTIVE_REFERENCE_SCALES["population_years"]
    mean = ensemble_critic_predictions(ensemble.task, ensemble.costs, dataset, rows, 4)
    # Members decode in float32 (the single-checkpoint path); the float64 mixture agrees to it.
    np.testing.assert_allclose(
        (mixture * values).sum(axis=1) * scale, mean["population_years"], rtol=1e-6
    )


@pytest.mark.parametrize("bounded", [True, False])
def test_ensemble_of_identical_checkpoints_equals_the_single_report(bounded: bool) -> None:
    task, costs = _models(1, bounded=bounded)
    twin = (copy.deepcopy(task), copy.deepcopy(costs))
    ensemble = validate_ensemble([_member(0, task, costs), _member(1, *twin)])
    dataset = _dataset()
    single = pe.critic_dataset_report(task, costs, dataset, 3, qol_reference=REFERENCE)
    averaged = pe.critic_dataset_report(
        ensemble.task, ensemble.costs, dataset, 3, qol_reference=REFERENCE
    )
    assert _dump(averaged) == _dump(single)


def test_ensemble_task_exposes_the_shared_configuration() -> None:
    task, costs = _models(1, support=LaborValueSupport())
    ensemble = validate_ensemble([_member(0, task, costs), _member(1, *_models(2))])
    assert isinstance(ensemble.task, EnsembleTask)
    assert ensemble.task.support == task.support and ensemble.task.qol_head == task.qol_head
    assert torch.equal(ensemble.task.bins, task.bins)
    assert ensemble.task.privileged_layout is None
    assert ensemble.task.is_bounded("qol_years") and not ensemble.task.is_bounded(
        "population_years"
    )


def test_mismatched_members_fail_closed() -> None:
    plain = _models(1)
    other = _models(2)
    privileged = _models(3, privileged=True)

    def check(second, match, **kwargs):
        with pytest.raises(ValueError, match=match):
            validate_ensemble([_member(0, *plain), _member(1, *second, **kwargs)])

    check(privileged, "input kind")
    check(other, "observation normalizer", tag="different")
    check(
        _models(2, support=LaborValueSupport(high=7.0)),
        "value support",
    )
    check(_models(2, bounded=False), "QoL head")
    check(_models(2, qol_head=LaborQoLHead(output="silu-clamp")), "QoL head")
    check(
        other,
        "loader corrections",
        corrections={"tail_normalize": "world-factor"},
    )
    check(other, "loader-correction source", source="explicit-override")


def test_privileged_members_need_one_privileged_normalizer() -> None:
    from test_labor_privileged_bridge import _normalizer

    same_a = _critics(_layout(True), _normalizer(0), seed=1)
    same_b = _critics(_layout(True), _normalizer(0), seed=2)
    ensemble = validate_ensemble([_member(0, *same_a), _member(1, *same_b)])
    assert ensemble.shared["input_kind"] == "privileged"
    assert (
        ensemble.shared["privileged_normalizer_digest"] == same_a[0].privileged_normalizer.digest()
    )
    refit = _critics(_layout(True), _normalizer(5), seed=4)
    with pytest.raises(ValueError, match="privileged normalizer"):
        validate_ensemble([_member(0, *same_a), _member(1, *refit)])


# ---------------------------------------------------------------------------
# The command: single-path invariant, ensemble report, member reports
# ---------------------------------------------------------------------------


@pytest.fixture()
def stubbed(tmp_path, monkeypatch):
    """Three synthetic checkpoints (files for hashing) behind stubbed model/data loaders."""
    models = {}
    for index, seed in enumerate((1, 2, 3)):
        path = tmp_path / f"critic{index}.pt"
        path.write_bytes(f"critic-{index}".encode())
        models[path] = _models(seed)
    twin = tmp_path / "twin0.pt"
    twin.write_bytes(b"critic-0")
    models[twin] = (
        copy.deepcopy(models[tmp_path / "critic0.pt"][0]),
        copy.deepcopy(models[tmp_path / "critic0.pt"][1]),
    )
    datasets = {"cal": _dataset(), "fit": _dataset(shift=10)}

    def load_models(path):
        task, costs = models[Path(path)]
        return (
            {"heuristic_critic_pretraining": {"dataset_digest": "fit", "epochs": 1}},
            None,
            task,
            costs,
            _Norm(),
        )

    def load_dataset(path, normalizer, *, require_privileged=False, corrections=None):
        return copy.deepcopy(datasets[str(path)])

    monkeypatch.setattr(pe, "_load_models", load_models)
    monkeypatch.setattr(pe, "_load_dataset", load_dataset)
    return tmp_path, list(models)


def _run(paths, **kwargs):
    return pe.run_critic_calibration(
        paths, "cal", kwargs.pop("fit", None), 3, qol_reference=REFERENCE, **kwargs
    )


def test_one_path_is_the_historical_report(stubbed) -> None:
    _, (a, *_rest) = stubbed
    bare = _run(a, fit="fit")
    listed = _run([a], fit="fit")
    assert _dump(listed) == _dump(bare)
    assert "ensemble" not in bare and "member_reports" not in bare
    assert bare["checkpoint"] == str(a)
    with pytest.raises(ValueError, match="member reports"):
        _run([a], member_reports=True)


def test_identical_checkpoints_reproduce_the_single_report(stubbed) -> None:
    _root, (a, _b, _c, twin) = stubbed
    single = _run(a, fit="fit")
    both = _run([a, twin], fit="fit")
    block = both.pop("ensemble")
    assert both.pop("checkpoint") == [str(a), str(twin)]
    single.pop("checkpoint")
    assert _dump(both) == _dump(single)
    assert block["member_count"] == 2
    assert [m["checkpoint"] for m in block["members"]] == [str(a), str(twin)]
    assert block["members"][0]["sha256"] == block["members"][1]["sha256"] != "0" * 64
    import hashlib

    assert block["members"][0]["sha256"] == hashlib.sha256(b"critic-0").hexdigest()
    assert "mean" in block["averaging"]["rule"] and "value" in block["averaging"]["qol_years"]
    assert block["validated_identical"]["input_kind"] == "plain"
    assert block["loader_corrections"] == {}


def test_member_reports_match_each_member_run_alone(stubbed) -> None:
    _root, (a, b, c, _twin) = stubbed
    report = _run([a, b, c], fit="fit", member_reports=True)
    assert [m["index"] for m in report["member_reports"]] == [0, 1, 2]
    for member, path in zip(report["member_reports"], (a, b, c), strict=True):
        alone = _run(path, fit="fit")
        assert member["checkpoint"] == str(path)
        assert _dump(member["calibration"]) == _dump(alone["calibration"])
        assert _dump(member["fitting"]) == _dump(alone["fitting"])
        assert _dump(member["generalisation_gap"]) == _dump(alone["generalisation_gap"])
        assert member["verdict"] == alone["verdict"]
    # The ensemble section is a different (averaged) evaluation than any member.
    assert _dump(report["calibration"]) != _dump(report["member_reports"][0]["calibration"])
    no_fit = _run([a, b], member_reports=True)
    assert "fitting" not in no_fit["member_reports"][0]
    text = pe.summarize_critic(report)
    assert "ENSEMBLE of 3" in text and "member 2" in text


def test_report_is_the_single_metrics_on_averaged_predictions(stubbed) -> None:
    _root, (a, b, _c, _twin) = stubbed
    report = _run([a, b])
    # Reference: the unchanged single-checkpoint metric code fed the hand-averaged values.
    from anthropocene_rl.autonomous_v7.labor_bc import critic_target_available, discounted_returns
    from anthropocene_rl.autonomous_v7.labor_model import LABOR_OBJECTIVE_REFERENCE_SCALES

    dataset = pe._load_dataset("cal", None)  # the stub's stored calibration data
    keep = critic_target_available(dataset).numpy().astype(bool)
    rows = np.flatnonzero(keep)
    (task_a, costs_a), (task_b, costs_b) = pe._load_models(a)[2:4], pe._load_models(b)[2:4]
    pa = pe.critic_predictions(task_a, costs_a, dataset, rows, 3)
    pb = pe.critic_predictions(task_b, costs_b, dataset, rows, 3)
    target = discounted_returns(dataset)["qol_years"].double().numpy()[rows]
    expected = pe.critic_head_metrics(
        (pa["qol_years"] + pb["qol_years"]) / 2.0,
        target,
        dataset["row_world"].numpy().reshape(-1)[rows],
        dataset["row_turn"].numpy().reshape(-1)[rows],
        LABOR_OBJECTIVE_REFERENCE_SCALES["qol_years"],
        zero_head=False,
    )
    got = report["calibration"]["heads"]["qol_years"]["all"]
    for key in ("pred_mean", "nmae", "bias"):
        assert got[key] == pytest.approx(expected[key], rel=1e-9)
    assert got["pred_mean"] != pytest.approx(
        pe.critic_head_metrics(
            pa["qol_years"],
            target,
            dataset["row_world"].numpy().reshape(-1)[rows],
            dataset["row_turn"].numpy().reshape(-1)[rows],
            LABOR_OBJECTIVE_REFERENCE_SCALES["qol_years"],
            zero_head=False,
        )["pred_mean"],
        rel=1e-9,
    )


def test_command_line_accepts_several_checkpoints() -> None:
    parser = pe.argument_parser()
    args = parser.parse_args(
        [
            "critic-calibration",
            "--checkpoint",
            "a.pt",
            "b.pt",
            "c.pt",
            "--dataset",
            "d1",
            "d2",
            "--member-reports",
            "--out",
            "o.json",
        ]
    )
    assert args.checkpoint == [Path("a.pt"), Path("b.pt"), Path("c.pt")]
    assert args.member_reports and args.dataset == [Path("d1"), Path("d2")]
    one = parser.parse_args(
        ["critic-calibration", "--checkpoint", "a.pt", "--dataset", "d", "--out", "o"]
    )
    assert one.checkpoint == [Path("a.pt")] and not one.member_reports


def test_mismatched_checkpoints_fail_closed_through_the_command(stubbed, monkeypatch) -> None:
    _root, (a, b, _c, _twin) = stubbed
    original = pe._load_models

    def mixed(path):
        state, actor, task, costs, norm = original(path)
        if Path(path) == b:
            norm = _Norm("other-normalizer")
        return state, actor, task, costs, norm

    monkeypatch.setattr(pe, "_load_models", mixed)
    with pytest.raises(ValueError, match="observation normalizer"):
        _run([a, b])
