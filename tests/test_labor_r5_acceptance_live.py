"""Live mechanics smoke of the r5 acceptance orchestration (bridge required).

Bounded local data (one fitting and one calibration world, 8 players, 2
decisions, truncated-zero targets under ``--smoke``), synthetic candidate
payloads carrying real root tensors, 2 arms x 2 seeds: initialize ->
pretrain-critic (stability on, exact resume state) -> A fit -> evaluate ->
determinism probe -> receipt. A smoke receipt can never PASS.
"""

from __future__ import annotations

import json

import pytest
import torch

from anthropocene_rl.autonomous_v7 import labor_advantage as la
from anthropocene_rl.autonomous_v7 import labor_r5_acceptance as ra
from anthropocene_rl.autonomous_v7 import labor_teacher_data as ltd
from anthropocene_rl.autonomous_v7.bridge import DEFAULT_BRIDGE
from anthropocene_rl.autonomous_v7.labor_model import BOUNDED_FEATURE_CONTRACT
from anthropocene_rl.autonomous_v7.labor_privileged import (
    LABOR_PRIVILEGED_LAYOUT_V1,
    SHARD_PREFIX,
    is_privileged_key,
)

pytestmark = pytest.mark.skipif(not DEFAULT_BRIDGE.is_file(), reason="model9 bridge is not built")


def _payloads(root, data, name, worlds, per_world):
    rows = int(data["row_world"].shape[0])
    shard = la.synthetic_candidate_shard(
        worlds=tuple(worlds),
        roots_per_world=per_world,
        candidates=5,
        seed=1,
        core_width=int(data["core"].shape[1]),
    )
    index = torch.arange(len(shard["roots"])) % rows
    shard["tensors"] = {
        k: v[index]
        for k, v in data.items()
        if isinstance(v, torch.Tensor)
        and not is_privileged_key(k)
        and not k.startswith(("target_", "label_", "reward_", "transition_", "row_", "tail_"))
    }
    shard["privileged"] = {
        k: v[index]
        for k, v in data.items()
        if isinstance(v, torch.Tensor) and k.startswith(SHARD_PREFIX)
    }
    out = root / name
    out.mkdir()
    torch.save(shard, out / "shard.pt")
    return out


def test_live_smoke_run_produces_a_held_receipt(tmp_path) -> None:
    from test_labor_bc import _fit_test_normalizer

    from anthropocene_rl.autonomous_v7.bridge import Model9Bridge
    from anthropocene_rl.autonomous_v7.labor_training import _normalization_digest
    from anthropocene_rl.autonomous_v7.labor_training import main as training_main
    from anthropocene_rl.autonomous_v7.model import load_checkpoint_normalizer

    with Model9Bridge() as bridge:
        if bridge.privileged_layout != LABOR_PRIVILEGED_LAYOUT_V1:
            pytest.skip("bridge does not emit privileged schema 1")
    norm = tmp_path / "norm.pt"
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
                    str(tmp_path / name),
                ]
            )
            == 0
        )
    pn = tmp_path / "pn.pt"
    training_main(
        [
            "fit-privileged-normalizer",
            "--dataset",
            str(tmp_path / "fit"),
            "--calibration-dataset",
            str(tmp_path / "cal"),
            "--out",
            str(pn),
        ]
    )
    normalizer = load_checkpoint_normalizer(
        torch.load(norm, weights_only=True), expected_feature_contract=BOUNDED_FEATURE_CONTRACT
    )
    data = ltd.load_labor_dataset(
        tmp_path / "fit",
        expected_normalizer_digest=_normalization_digest(normalizer),
        require_privileged=True,
        training=False,
    )
    fit_payloads = _payloads(tmp_path, data, "fit-payloads", range(90100, 90106), 3)
    reserved = _payloads(tmp_path, data, "reserved-payloads", la.RESERVED_HELDOUT_WORLDS, 2)
    out = tmp_path / "run"
    argv = [
        "run",
        "--normalizer-checkpoint",
        str(norm),
        "--privileged-normalizer",
        str(pn),
        "--fit-data",
        str(tmp_path / "fit"),
        "--calibration-data",
        str(tmp_path / "cal"),
        "--fit-payloads",
        str(fit_payloads),
        "--reserved-payloads",
        str(reserved),
        "--seeds",
        "0,1",
        "--required-seeds",
        "2",
        "--local",
        "--smoke",
        "--advantage-epochs",
        "2",
        "--determinism-steps",
        "2",
        "--stability-every",
        "1",
        "--resume-every",
        "1",
        "--batch-size",
        "8",
        "--jobs",
        "2",
        "--threads",
        "1",
        "--attempt-id",
        "live-smoke",
        "--out",
        str(out),
    ]
    assert ra.main(argv) == 0
    receipt = json.loads((out / "gate-receipt.json").read_text())
    assert receipt["machine_verdict"] in ("HOLD", "FAIL") and receipt["verdict_basis"]["smoke"]
    assert receipt["determinism"]["status"] == "PASS"
    comparison = receipt["arm_comparisons"]["privileged-e4-vs-control-e4"]
    assert comparison["comparability"]["status"] == "PASS"
    for arm in ("privileged-e4", "control-e4"):
        for seed in (0, 1):
            work = out / arm / f"seed-{seed}"
            evidence = json.loads((work / "evidence.json").read_text())
            critic = evidence["stability"]["critic"]
            assert critic["checks"]["finite_updates"]["status"] == "PASS"
            assert critic["checks"]["frozen_gradients_zero"]["status"] == "PASS"
            expected = "PASS" if arm.startswith("privileged") else "NOT_APPLICABLE"
            assert critic["checks"]["privileged_attention"]["status"] == expected
            assert (
                evidence["stability"]["advantage"]["checks"]["frozen_gradients_zero"]["status"]
                == "PASS"
            )
            assert (work / "stability-critic.jsonl").stat().st_size > 0
            assert (work / "critic.resume.pt").exists()
    # Rerunning is a no-op on finished stages (resumable) and reproduces the verdict.
    assert ra.main(argv) == 0
    again = json.loads((out / "gate-receipt.json").read_text())
    assert again["machine_verdict"] == receipt["machine_verdict"]
    assert (out / "evidence-manifest.json").exists()
