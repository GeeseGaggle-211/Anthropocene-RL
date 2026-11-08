"""labor_a_measure: train --a-mode gae iteration 0, measure-only (synthetic CPU fixtures).

Reuses WP-A1's fixtures (fake bridge, real actor/critics, random A head); no native DLL,
no sealed (3000-3059) or reserved (90154-90159) worlds.
"""

from __future__ import annotations

import json

import pytest
import torch
from test_labor_a_ppo_wiring import _train_args, e2e  # noqa: F401  (pytest fixture)

from anthropocene_rl.autonomous_v7 import labor_a_measure as measure
from anthropocene_rl.autonomous_v7 import labor_a_ppo as ap
from anthropocene_rl.autonomous_v7 import labor_training as training
from anthropocene_rl.autonomous_v7.labor_ppo import INITIAL_POLICY_HEADS


def _measure_args(e2e, out, *extra):  # noqa: F811
    checkpoint, rollout, _anchor, _gate = e2e["files0"]
    return ["--checkpoint", str(checkpoint), "--rollout", str(rollout),
            "--a-warm-start", str(e2e["warm"]), "--a-critic", str(e2e["critic"]),
            "--out", str(out), "--epochs", "2", "--batch-size", "2", "--actor-lr", "1e-6",
            "--threads", str(torch.get_num_threads()), *extra]


def test_measure_equals_train_gae_iteration_zero_without_saving(e2e):  # noqa: F811
    checkpoint, rollout, anchor, gate = e2e["files0"]
    tmp = e2e["tmp"]
    before = {p: p.read_bytes() for p in (checkpoint, rollout, e2e["warm"])}
    files_before = set(tmp.rglob("*"))

    out = tmp / "d7.json"
    assert measure.main(_measure_args(e2e, out)) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    out_report = tmp / "d7-report-gate.json"
    measure.main(_measure_args(e2e, out_report, "--calibration-report", str(gate)))
    with_report = json.loads(out_report.read_text(encoding="utf-8"))

    # Nothing but the JSON reports was written; the inputs are untouched.
    assert set(tmp.rglob("*")) - files_before == {out, out_report}
    assert all(p.read_bytes() == data for p, data in before.items())

    # The real train --a-mode gae (with its critic update and dry run) on the same files.
    gae = ("--a-mode", "gae", "--a-warm-start", str(e2e["warm"]),
           "--a-critic", str(e2e["critic"]), "--a-telemetry-dry-run")
    trained = tmp / "it1.pt"
    training.main(_train_args(checkpoint, rollout, anchor, gate, trained, *gae))
    record = torch.load(trained, weights_only=True)["last_update"]["aControlVariate"]

    assert result["eta"]["used"] == ap.zero_eta() == record["etaUsed"]
    assert result["eta"]["source"] == "first-gae-iteration-measure-only" == record["etaSource"]
    assert result["eta"]["next"] == record["nextEta"]
    assert set(result["d7"]) == set(INITIAL_POLICY_HEADS)
    # Same functions, rows and torch thread count as train: bitwise equal.
    for head in INITIAL_POLICY_HEADS:
        for key in ("varianceVOnly", "varianceControlVariate", "reduction", "admit", "aBlind"):
            assert result["d7"][head][key] == record["d7"][head][key], (head, key)
    assert result["d7"]["trade_value"]["aBlind"] and not result["d7"]["trade_value"]["admit"]
    assert result["frozen"]["criticDigest"] == record["frozen"]["criticDigest"]
    assert result["frozen"]["unchangedAfterIteration"] is True
    assert result["counts"] == record["counts"]
    # The dry-run KL does not read the critic: equal to train's, gate kind irrelevant.
    kl = record["warmup"]["dryRunKL"]
    assert len(kl) == 2 and result["dryRunKL"]["perEpoch"] == kl
    assert with_report["dryRunKL"]["perEpoch"] == result["dryRunKL"]["perEpoch"]
    assert result["config"]["dryRun"]["gate"] == "measurement-only-binding"
    assert with_report["config"]["dryRun"]["gate"] == "calibration-report"
    # Warm-up telemetry is reported per component with the UNCHANGED checkpoint critic.
    warmup = result["warmup"]
    assert warmup["status"] == "measured"
    assert warmup["dryRunKL"] == result["dryRunKL"]["perEpoch"]
    assert {"qol_years", "population_years", "feasibility", "settlement_guard"} <= set(
        warmup["components"]
    )
    assert "decision" in warmup
    assert result["rollout"]["parents"] == 5 and result["rollout"]["retries"] == 2
    assert result["rollout"]["parentsPerWorld"] == {"50000": 2, "50001": 2, "50002": 1}
    assert result["rollout"]["resetOptions"] == training.LEGACY_COLLECT_RESET.state_dict()
    assert result["rollout"]["resetOptionsSource"] == "legacy-default-bridge.reset(seed)"
    assert result["storedStateTrainWouldWrite"]["measuredIteration"] == 0
    assert {"load", "setup", "prepareAndD7", "dryRunKL", "finish", "total"} <= set(
        result["seconds"]
    )


def test_measure_no_dry_run_and_d7_rows(e2e):  # noqa: F811
    out = e2e["tmp"] / "d7-nodry.json"
    measure.main(_measure_args(e2e, out, "--no-dry-run", "--d7-rows", "3"))
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["dryRunKL"]["perEpoch"] is None
    assert result["warmup"]["dryRunKL"] == "not measured"
    assert all(v["rows"] == 3 for v in result["d7"].values())
    assert result["config"]["d7Rows"] == 3


def test_measure_refuses_a_rollout_of_another_checkpoint(e2e):  # noqa: F811
    _checkpoint, rollout, _anchor, _gate = e2e["files0"]
    recorded = torch.load(rollout, weights_only=True)
    recorded["metadata"]["sourceCheckpointSha256"] = "0" * 64
    other = e2e["tmp"] / "other-rollout.pt"
    torch.save(recorded, other)
    args = _measure_args(e2e, e2e["tmp"] / "x.json")
    args[args.index("--rollout") + 1] = str(other)
    with pytest.raises(ValueError, match="belongs to another checkpoint"):
        measure.main(args)
