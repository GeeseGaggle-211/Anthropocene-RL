"""labor_a_measure: train --a-mode gae iteration 0, measure-only (synthetic CPU fixtures).

Reuses WP-A1's fixtures (fake bridge, real actor/critics, random A head); no native DLL,
no sealed (3000-3059) or reserved (90154-90159) worlds.
"""

from __future__ import annotations

import json

import pytest
import torch
from test_labor_a_ppo_wiring import _e2e_rows, _train_args, e2e  # noqa: F401  (pytest fixture)

from anthropocene_rl.autonomous_v7 import labor_a_measure as measure
from anthropocene_rl.autonomous_v7 import labor_a_ppo as ap
from anthropocene_rl.autonomous_v7 import labor_training as training
from anthropocene_rl.autonomous_v7.labor_actor_optim import actor_optimizer_state_by_name
from anthropocene_rl.autonomous_v7.labor_ppo import INITIAL_POLICY_HEADS
from anthropocene_rl.autonomous_v7.labor_value_calibration import labor_critic_digest
from anthropocene_rl.autonomous_v7.provenance import checkpoint_hash


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


# ---------------------------------------------------------------------------
# --apply: train --a-mode off's actor update, actor-only, written to a checkpoint
# ---------------------------------------------------------------------------

CRITIC_FIELDS = ("task_critic", "cost_critics", "critic_optimizer",
                 "critic_training_world_seeds", "critic_training_independent_unit_ids")
REWRITTEN = {"actor", "actor_optimizer", "iteration", "sampling_generator_state",
             "shuffle_rng_state", "last_update", measure.STEP_CHECKS_KEY}


def _apply_args(checkpoint, rollout, out_checkpoint, report, *extra):
    # train's fixture flags (_train_args): 2 epochs of 2-parent minibatches, lr 1e-6.
    return ["--apply", "--checkpoint", str(checkpoint), "--rollout", str(rollout),
            "--out-checkpoint", str(out_checkpoint), "--out", str(report),
            "--epochs", "2", "--batch-size", "2", "--actor-lr", "1e-6",
            "--threads", str(torch.get_num_threads()), *extra]


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _report(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_from(monkeypatch, tmp, checkpoint, tag):
    """The real collect command on ``checkpoint`` (fake bridge rows sampled by its actor)."""
    _, actor, task, costs, _ = training._models_from_checkpoint(_load(checkpoint))
    rows, bootstrap = _e2e_rows(monkeypatch, (actor, task, costs))
    monkeypatch.setattr(training, "collect_labor_rollout", lambda **kw: (rows, bootstrap))
    rollout = tmp / f"{tag}-rollout.pt"
    training.main(["collect", "--checkpoint", str(checkpoint), "--worlds",
                   "50000,50001,50002", "--turns", "2", "--out", str(rollout)])
    return rollout


def _critic_digest(state):
    _, _, task, costs, _ = training._models_from_checkpoint(state)
    return labor_critic_digest(task, costs)


def _assert_only_actor_changed(state, source):
    for key in CRITIC_FIELDS:  # the critic, its optimizer and its data history
        assert measure._identical(state[key], source[key]), key
    assert set(state) - REWRITTEN == set(source) - REWRITTEN
    for key in set(source) - REWRITTEN:
        assert measure._identical(state[key], source[key]), key
    assert _critic_digest(state) == _critic_digest(source)


def test_apply_is_train_off_actor_update_and_leaves_the_critic_bitwise(e2e):  # noqa: F811
    checkpoint, rollout, anchor, gate = e2e["files0"]
    tmp = e2e["tmp"]
    inputs = {p: p.read_bytes() for p in (checkpoint, rollout)}
    source = _load(checkpoint)

    out, report_path = tmp / "apply.pt", tmp / "apply.json"
    assert measure.main(_apply_args(checkpoint, rollout, out, report_path,
                                    "--target-kl", "10")) == 0
    report = _report(report_path)
    applied = _load(out)
    # the same update behind the calibration-report gate: identical bits
    out_gate = tmp / "apply-gate.pt"
    measure.main(_apply_args(checkpoint, rollout, out_gate, tmp / "apply-gate.json",
                             "--target-kl", "10", "--calibration-report", str(gate)))
    applied_gate = _load(out_gate)
    assert all(p.read_bytes() == data for p, data in inputs.items())

    # The real train (--a-mode off, the default) on the same files and flags.
    trained_path = tmp / "train-off.pt"
    training.main(_train_args(checkpoint, rollout, anchor, gate, trained_path))
    trained = _load(trained_path)
    assert trained["last_update"]["actor"]["acceptedSteps"] > 0
    assert not measure._identical(trained["task_critic"], source["task_critic"])

    _, actor, *_ = training._models_from_checkpoint(applied)
    for state in (applied, applied_gate):
        # the actor, its Adam state and the progress fields are train's, bit for bit
        for key in ("actor", "iteration", "sampling_generator_state", "shuffle_rng_state"):
            assert measure._identical(state[key], trained[key]), key
        assert measure._identical(
            actor_optimizer_state_by_name(actor, state["actor_optimizer"]),
            actor_optimizer_state_by_name(actor, trained["actor_optimizer"]),
        )
        assert state["last_update"]["actor"]["losses"] == trained["last_update"]["actor"]["losses"]
        _assert_only_actor_changed(state, source)
    assert any(not torch.equal(v, source["actor"][k]) for k, v in applied["actor"].items())

    assert report["criticUpdated"] is False
    assert applied["last_update"]["criticUpdated"] is False
    (provenance,) = applied[measure.STEP_CHECKS_KEY]
    assert provenance["criticUpdated"] is False and provenance["criticOptimizerUpdated"] is False
    assert provenance["inputCheckpointSha256"] == checkpoint_hash(checkpoint)
    assert provenance["rolloutSha256"] == checkpoint_hash(rollout)
    assert provenance["flags"]["targetKL"] == 10.0 and provenance["flags"]["aMode"] == "off"
    assert provenance["codeCommit"]
    assert report["inputs"]["onPolicy"] is True
    assert report["output"]["checkpointSha256"] == checkpoint_hash(out)
    update = report["update"]
    assert update["attemptedSteps"] == update["acceptedSteps"] == 6  # 2 epochs x 3 minibatches
    assert update["rolledBackSteps"] == 0 and not update["rollback"]
    assert not update["stoppedEarly"]
    assert len(update["perStepAuditKL"]) == 6 and len(update["perEpoch"]) == 2
    assert [e["auditKLAfterLastAcceptedStep"] for e in update["perEpoch"]] == [
        update["perStepAuditKL"][2], update["perStepAuditKL"][5]]
    assert update["lrFactorPerAttemptedStep"] == [1.0] * 6
    assert report["adam"]["stepsBefore"] == 0 and report["adam"]["stepsAfter"] == 6
    deltas = report["logProbs"]
    assert set(deltas["afterVsBefore"]) == {*INITIAL_POLICY_HEADS, "initialJoint", "retryJoint"}
    assert deltas["afterVsBefore"]["initialJoint"]["meanAbs"] > 0
    assert deltas["afterVsBefore"]["retryJoint"]["rows"] == 2
    # on-policy: the checkpoint actor reproduces the stored behaviour log-probs
    assert all(v["maxAbs"] < 1e-4 for v in deltas["beforeVsBehaviour"].values())


def test_apply_with_forced_rollback_leaves_the_actor_bitwise(e2e):  # noqa: F811
    checkpoint, rollout, _anchor, _gate = e2e["files0"]
    tmp = e2e["tmp"]
    source = _load(checkpoint)
    out, report_path = tmp / "rolled-back.pt", tmp / "rolled-back.json"
    args = _apply_args(checkpoint, rollout, out, report_path, "--target-kl", "1e-30")
    args[args.index("--actor-lr") + 1] = "1e-2"  # a step large enough for a KL > 0
    measure.main(args)
    report = _report(report_path)
    applied = _load(out)
    update = report["update"]
    assert update["rollback"] and update["stoppedEarly"]
    assert update["attemptedSteps"] == 1 and update["acceptedSteps"] == 0
    assert update["rolledBackSteps"] == 1 and len(update["perStepAuditKL"]) == 1
    assert update["perEpoch"][0]["rejectedStepAuditKL"] == update["perStepAuditKL"][0] > 1e-30
    assert update["perEpoch"][0]["auditKLAfterLastAcceptedStep"] is None
    assert not update["actorChanged"]
    assert measure._identical(applied["actor"], source["actor"])
    _, actor, *_ = training._models_from_checkpoint(applied)
    assert measure._identical(
        actor_optimizer_state_by_name(actor, applied["actor_optimizer"]),
        actor_optimizer_state_by_name(actor, source["actor_optimizer"]),
    )
    assert report["adam"]["stepsBefore"] == report["adam"]["stepsAfter"] == 0
    assert all(v["maxAbs"] == 0 for v in report["logProbs"]["afterVsBefore"].values())
    assert (report["output"]["behaviorPolicyDigest"]
            == report["inputs"]["checkpointBehaviorPolicyDigest"])
    _assert_only_actor_changed(applied, source)


def test_apply_output_round_trips_through_collect_and_a_second_apply(
    e2e, monkeypatch  # noqa: F811
):
    checkpoint, rollout, _anchor, _gate = e2e["files0"]
    tmp = e2e["tmp"]
    source = _load(checkpoint)
    first = tmp / "step-1.pt"
    measure.main(_apply_args(checkpoint, rollout, first, tmp / "step-1.json",
                             "--actor-lr-warmup-steps", "12", "--target-kl", "10"))
    report1 = _report(tmp / "step-1.json")
    assert report1["update"]["lrFactorPerAttemptedStep"] == [(n + 1) / 12 for n in range(6)]

    # collect accepts the checkpoint and binds its rollout to the UPDATED actor
    rollout1 = _collect_from(monkeypatch, tmp, first, "step-1")
    metadata = _load(rollout1)["metadata"]
    assert metadata["sourceCheckpointSha256"] == checkpoint_hash(first)
    assert metadata["behaviorPolicyDigest"] == report1["output"]["behaviorPolicyDigest"]
    assert metadata["behaviorPolicyDigest"] != report1["inputs"]["checkpointBehaviorPolicyDigest"]
    # its sampling stream continues where the input rollout's ended (as after train)
    assert torch.equal(metadata["samplingGeneratorStateBefore"],
                       _load(rollout)["metadata"]["samplingGeneratorStateAfter"])

    # the old rollout is refused against the new checkpoint
    with pytest.raises(ValueError, match="belongs to another checkpoint"):
        measure.main(_apply_args(first, rollout, tmp / "x.pt", tmp / "x.json"))
    assert not (tmp / "x.pt").exists()

    # a second apply continues the Adam moments, the warm-up count and the streams
    second = tmp / "step-2.pt"
    measure.main(_apply_args(first, rollout1, second, tmp / "step-2.json",
                             "--actor-lr-warmup-steps", "12", "--target-kl", "10"))
    report2 = _report(tmp / "step-2.json")
    assert report2["inputs"]["onPolicy"] is True
    assert report2["adam"]["stepsBefore"] == report1["adam"]["stepsAfter"] == 6
    assert report2["adam"]["stepsAfter"] == 12
    assert report2["update"]["lrFactorPerAttemptedStep"] == [(n + 7) / 12 for n in range(6)]
    assert report2["inputs"]["iteration"] == 1 and report2["output"]["iteration"] == 2
    state2 = _load(second)
    assert [c["inputCheckpointSha256"] for c in state2[measure.STEP_CHECKS_KEY]] == [
        checkpoint_hash(checkpoint), checkpoint_hash(first)]
    _assert_only_actor_changed(state2, source)


def test_apply_refusals(e2e):  # noqa: F811
    checkpoint, rollout, _anchor, _gate = e2e["files0"]
    tmp = e2e["tmp"]
    existing = tmp / "exists.pt"
    existing.write_bytes(b"keep")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        measure.main(_apply_args(checkpoint, rollout, existing, tmp / "r.json"))
    assert existing.read_bytes() == b"keep"
    with pytest.raises(SystemExit):  # A arguments belong to measure mode
        measure.main(_apply_args(checkpoint, rollout, tmp / "n.pt", tmp / "r.json",
                                 "--a-critic", str(checkpoint)))
    with pytest.raises(SystemExit):  # --target-kl belongs to apply mode
        measure.main(_measure_args(e2e, tmp / "m.json", "--target-kl", "0.02"))
    with pytest.raises(SystemExit):  # --apply needs --out-checkpoint
        measure.main(["--apply", "--checkpoint", str(checkpoint), "--rollout", str(rollout),
                      "--out", str(tmp / "r.json")])
