"""Several teacher fitting / calibration datasets (r5 collection batches) read as one.

``labor_teacher_data.merge_labor_datasets`` pools the batches' shards in the order one
published dataset stores them, so batches that partition a set of worlds give the same
tensors, and the same critic fit bit for bit, as one dataset over all of them. One
dataset is read exactly as before.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import torch
from test_labor_model import _labor_tensors
from test_labor_privileged_model import _layout

from anthropocene_rl.autonomous_v7 import labor_advantage as la
from anthropocene_rl.autonomous_v7 import labor_bc
from anthropocene_rl.autonomous_v7 import labor_r5_acceptance as ra
from anthropocene_rl.autonomous_v7 import labor_teacher_data as ltd
from anthropocene_rl.autonomous_v7.labor_bc import pretrain_critic_on_dataset
from anthropocene_rl.autonomous_v7.labor_model import (
    LaborCostCritics,
    LaborQoLHead,
    LaborTaskCritic,
)
from anthropocene_rl.autonomous_v7.provenance import digest

WORLDS = (90100, 90101, 90102, 90103)
TAIL_RECORD = {
    "method": labor_bc.RETURN_TAIL_RANDOMIZED,
    "estimator": "russian-roulette",
    "draws": 16,
    "rho": 0.95,
    "cap": 200,
    "schedule_seed_base": 7,
    "prefix_terms": 3,
}


def _arrays(world: int, *, randomized: bool) -> dict[str, np.ndarray]:
    """One world's shard: 2 camps x 3 decisions, content a function of the world only."""
    torch.manual_seed(world)
    rows = 6
    arrays = {k: v.numpy() for k, v in _labor_tensors(batch=rows).items()}
    rng = np.random.default_rng(world)
    arrays["core"] = rng.standard_normal((rows, 10)).astype(np.float32)
    arrays.update(
        row_world=np.full(rows, world, dtype=np.int64),
        row_camp=np.array([1, 1, 1, 2, 2, 2], dtype=np.int64),
        row_turn=np.array([0, 1, 2, 0, 1, 2], dtype=np.int64),
        row_attempt=np.zeros(rows, dtype=np.int64),
        row_quarantined=np.zeros(rows, dtype=bool),
        reward_qol_years=rng.uniform(5.0, 15.0, rows),
        reward_population_years=rng.uniform(0.5, 2.0, rows),
        reward_feasibility=np.zeros(rows),
        reward_settlement_guard=np.zeros(rows),
        transition_elapsed_years=np.full(rows, 20.0),
        transition_terminal=np.array([False, False, False] * 2)
        if randomized
        else np.array([False, False, True] * 2),
    )
    if randomized:
        arrays[labor_bc.TAIL_METHOD_FIELD] = np.full(rows, labor_bc.TAIL_RANDOMIZED, np.int64)
        arrays[labor_bc.TAIL_CUTOFF_TURN_FIELD] = np.full(rows, 3, np.int64)
        for head in labor_bc.TAIL_HEADS:
            per_camp = rng.uniform(1.0, 3.0, 2)
            arrays[labor_bc.TAIL_VALUE_PREFIX + head] = np.repeat(per_camp, 3)
            arrays[labor_bc.TAIL_STANDARD_ERROR_PREFIX + head] = np.full(rows, 0.1)
    return arrays


def _publish(
    directory: Path,
    worlds,
    *,
    randomized: bool = True,
    batch: str = "single",
    normalizer: str = "n",
    turns: int = 3,
    partition: str | None = None,
) -> Path:
    options = {"turns": turns, "reset_options": {"players": 8}, "batch": batch}
    if partition is not None:
        options["partition"] = partition
    body = {"schema": ltd.SCHEMA, "source_kind": ltd.SOURCE_KIND, "options": options}
    source = dict(body, source_run=digest(body))
    directory.mkdir(parents=True)
    shards = []
    for world in worlds:
        meta = {
            "source_run": source["source_run"],
            "world_seed": world,
            "normalizer_digest": normalizer,
            "counters": {"total_decisions": 6, "initial_accepted": 6},
            "label_unavailable_reasons": {},
            "quarantine": [],
        }
        if randomized:
            meta["return_tail"] = labor_bc.RETURN_TAIL_RANDOMIZED
            meta["return_tail_record"] = dict(TAIL_RECORD)
        else:
            meta["return_tail"] = labor_bc.RETURN_TAIL_TRUNCATED_ZERO
        if partition is not None:
            meta["partition"] = partition
        shard = directory / f"world-{world}"
        record = ltd.write_shard(shard, _arrays(world, randomized=randomized), meta)
        shards.append(
            {
                "path": f"world-{world}",
                "world_seed": world,
                "meta_sha256": ltd.checkpoint_hash(shard / "meta.json"),
                "data_digest": record["data_digest"],
            }
        )
    ltd.publish_dataset(directory, source, shards)
    return directory


@pytest.fixture
def batches(tmp_path):
    single = _publish(tmp_path / "all", WORLDS)
    b01 = _publish(tmp_path / "fit8-b01", (90100, 90102), batch="b01")
    b02 = _publish(tmp_path / "fit8-b02", (90101, 90103), batch="b02")
    return single, b01, b02


PROVENANCE = {
    "dataset_digest",
    "source_run",
    "component_dataset_digests",
    "component_directories",
    "component_source_runs",
}


def _assert_same_dataset(merged, single) -> None:
    tensors = {k for k, v in single.items() if isinstance(v, torch.Tensor)}
    assert tensors == {k for k, v in merged.items() if isinstance(v, torch.Tensor)}
    for key in tensors:
        assert merged[key].dtype == single[key].dtype, key
        assert torch.equal(merged[key], single[key]), key
    common = {k: v for k, v in single["collection"].items() if k not in PROVENANCE}
    assert {k: merged["collection"][k] for k in common} == common


def test_batches_partitioning_the_worlds_read_as_the_one_dataset(batches) -> None:
    single, b01, b02 = batches
    one = ltd.load_labor_dataset(single)
    for listing in ([b01, b02], [b02, b01]):  # listing order does not matter
        merged = ltd.merge_labor_datasets(listing)
        _assert_same_dataset(merged, one)
        assert merged["collection"]["world_seeds"] == list(WORLDS)
        assert merged["collection"]["return_tail"] == labor_bc.RETURN_TAIL_RANDOMIZED
        assert set(merged["collection"]["return_tail_by_world"]) == {str(w) for w in WORLDS}
    assert (
        ltd.merge_labor_datasets([b01, b02])["collection"]["dataset_digest"]
        == (ltd.merge_labor_datasets([b02, b01])["collection"]["dataset_digest"])
    )
    # Equal mass per world over the combined set, as one multi-world dataset.
    merged = ltd.merge_labor_datasets([b02, b01])
    for world in WORLDS:
        mass = float(merged["row_weight"][merged["row_world"] == world].sum())
        assert mass == pytest.approx(len(merged["row_world"]) / len(WORLDS))
    # The loaders: one directory is load_labor_dataset itself; several teacher batches
    # merge (critic path), and stay per-directory aggregated parts without merge_teacher.
    alone = ltd.load_labor_datasets([single], merge_teacher=True)
    _assert_same_dataset(alone, one)
    assert alone["collection"] == one["collection"]
    _assert_same_dataset(ltd.load_labor_datasets([b01, b02], merge_teacher=True), one)
    assert ltd.load_labor_datasets([b01, b02])["collection"]["source_kind"] == "aggregated"
    _assert_same_dataset(ltd.load_labor_data([b02, b01], training=False), one)
    assert ltd.load_labor_data(single)["collection"] == one["collection"]


def _fit(dataset, *, schedule=None):
    torch.manual_seed(0)
    task = LaborTaskCritic(
        _layout(False), width=16, token_width=16, qol_head=LaborQoLHead(target="td-lambda")
    )
    costs = LaborCostCritics(_layout(False), width=16, token_width=16)
    summary = pretrain_critic_on_dataset(
        task_critic=task,
        cost_critics=costs,
        dataset=dataset,
        epochs=2,
        batch_size=4,
        rng=np.random.default_rng(11),
        require_randomized_tail=True,
        family_schedule=schedule,
    )
    state = {**task.state_dict(), **{f"c.{k}": v for k, v in costs.state_dict().items()}}
    return summary, state


def _same_state(a, b) -> bool:
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def test_critic_fit_on_merged_batches_is_bitwise_the_single_dataset_fit(batches) -> None:
    from anthropocene_rl.autonomous_v7.labor_training import _load_labor_teacher_dataset

    single, b01, b02 = batches

    def loaded(directories, weights=None):
        args = argparse.Namespace(
            dataset=list(directories),
            learner_state_continuations=False,
            curated_scenarios=False,
            drop_reserved_worlds=False,
            family_weights=weights,
        )
        return _load_labor_teacher_dataset(args, normalizer=None)

    from unittest import mock

    with mock.patch(
        "anthropocene_rl.autonomous_v7.labor_training._normalization_digest", lambda _: "n"
    ):
        # One dataset: the loader is exactly load_labor_dataset (current behaviour).
        one = loaded([single])
        reference = ltd.load_labor_dataset(single)
        assert one["collection"] == reference["collection"]
        assert _same_state(_fit(one)[1], _fit(reference)[1])
        # TD(lambda) refresh and randomized-tail targets: merged == single, bitwise.
        summary_one, state_one = _fit(one)
        summary_merged, state_merged = _fit(loaded([b02, b01]))
        assert _same_state(state_one, state_merged)
        for key in ("final_loss", "rows", "world_seeds", "return_tail", "return_tail_settings"):
            assert summary_one[key] == summary_merged[key], key
        # Family weights with a decaying schedule, groups (source, world): same equality.
        from anthropocene_rl.autonomous_v7.labor_scenarios import continued_schedule

        weights = {"small-camp": 2.0, "farming-unlocked": 3.0}
        schedule = continued_schedule(weights, "cosine", None, 2, None)
        weighted_one = loaded([single], weights="small-camp=2,farming-unlocked=3")
        weighted_merged = loaded([b01, b02], weights="small-camp=2,farming-unlocked=3")
        for key in ("row_weight", "row_weight_unweighted", "row_family_tags"):
            assert torch.equal(weighted_one[key], weighted_merged[key]), key
        assert (
            weighted_one["collection"]["family_weighting"]
            == weighted_merged["collection"]["family_weighting"]
        )
        assert _same_state(
            _fit(weighted_one, schedule=schedule)[1], _fit(weighted_merged, schedule=schedule)[1]
        )


def test_merge_refuses_inconsistent_batches(tmp_path, batches) -> None:
    single, b01, _ = batches
    with pytest.raises(ltd.ProvenanceError, match="more than once"):
        ltd.merge_labor_datasets([b01, b01])
    with pytest.raises(ltd.ProvenanceError, match="two merged"):
        ltd.merge_labor_datasets([single, b01])
    other_norm = _publish(tmp_path / "norm", (90104,), batch="n2", normalizer="other")
    with pytest.raises(ltd.ProvenanceError, match="normalizers"):
        ltd.merge_labor_datasets([b01, other_norm])
    turns = _publish(tmp_path / "turns", (90105,), batch="t", turns=4)
    with pytest.raises(ltd.ProvenanceError, match="horizons"):
        ltd.merge_labor_datasets([b01, turns])
    truncated = _publish(tmp_path / "trunc", (90106,), batch="z", randomized=False)
    with pytest.raises(ltd.ProvenanceError, match=r"arrays|return tails"):
        ltd.merge_labor_datasets([b01, truncated])
    reserved = _publish(tmp_path / "reserved", (90154,), batch="r")
    with pytest.raises(ValueError, match="reserved"):
        ltd.merge_labor_datasets([b01, reserved])
    heldout = _publish(tmp_path / "heldout", (90107,), batch="h", partition="held-out-test")
    with pytest.raises(ValueError, match="held-out"):
        ltd.merge_labor_datasets([b01, heldout])
    # Evaluation readers (training=False) still refuse mixing partitions.
    with pytest.raises(ltd.ProvenanceError, match="partitions"):
        ltd.merge_labor_datasets([b01, heldout], training=False)
    # Calibration worlds never fit a critic, whichever batch carries them.
    calibration = _publish(tmp_path / "cal", (90180,), batch="c")
    merged = ltd.merge_labor_datasets([b01, calibration])
    with pytest.raises(ValueError, match="calibration"):
        _fit(merged)


def test_candidate_payloads_from_several_batches_and_repeats_refused(tmp_path) -> None:
    paths = []
    for index, worlds in enumerate(((90100, 90101), (90102, 90103))):
        shard = la.synthetic_candidate_shard(
            worlds=worlds, roots_per_world=2, candidates=3, seed=index
        )
        directory = tmp_path / f"payload-b0{index + 1}"
        directory.mkdir()
        torch.save(shard, directory / "advantage-candidates.pt")
        paths.append(directory)
    both = la.load_candidate_shards(paths)
    assert both.worlds == [90100, 90101, 90102, 90103]
    with pytest.raises(ValueError, match="more than once"):
        la.load_candidate_shards([paths[0], paths[0] / "advantage-candidates.pt"])


def test_acceptance_cli_takes_several_fit_and_calibration_datasets() -> None:
    parser = ra.argument_parser()
    evaluate = parser.parse_args(
        [
            "evaluate",
            "--arm",
            "a",
            "--seed",
            "0",
            "--checkpoint",
            "c",
            "--advantage",
            "a",
            "--calibration-data",
            "cal8-b01",
            "cal8-b02",
            "cal8-b03",
            "--fit-data",
            "f1",
            "f2",
            "--reserved-payloads",
            "r",
            "--out",
            "o",
        ]
    )
    assert [str(p) for p in evaluate.calibration_data] == ["cal8-b01", "cal8-b02", "cal8-b03"]
    assert [str(p) for p in evaluate.fit_data] == ["f1", "f2"]
    args = argparse.Namespace(
        fit_data=[Path("f1"), Path("f2")],
        learner_state_continuations=False,
        curated_scenarios=False,
    )
    assert ra._evaluation_fit_data(args) == [Path("f1"), Path("f2")]
    args.learner_state_continuations = True
    assert ra._evaluation_fit_data(args) == [Path("f1")]
    script = (ra.REPOSITORY / "tools" / "r5_critic_acceptance.sh").read_text(encoding="utf-8")
    assert 'split_list "$FIT_DATA" fit_data' in script
    assert '--calibration-data "${cal_data[@]}"' in script
