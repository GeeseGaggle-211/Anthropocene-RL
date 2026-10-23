from __future__ import annotations

import copy
import sys

import pytest
import torch

from anthropocene_rl.autonomous_v7 import economic_trace_training_smoke as smoke


def payload():
    initial = {name: {"weight": torch.tensor([0.0])} for name in ("actor", "critic", "cost")}
    final = {name: {"weight": torch.tensor([1.0])} for name in ("actor", "critic", "cost")}
    return {
        "initial": initial,
        "final": final,
        "optimizers": [{"step": torch.tensor(1)}],
        "rng": {"torch": torch.tensor([1], dtype=torch.uint8)},
        "rollout": {
            "digest": "same",
            "initialObservationDigest": "same",
            "actionDigest": "same",
            "rewardDigest": "same",
            "rewardSum": 1.0,
            "decisions": 16,
            "producerMetadataDigest": "producer",
        },
        "stats": {"decisions": 16.0, "worker_contracts": [{"economic": False}]},
        "lagrange": 0.1,
        "configuration": {"critic": "scalar"},
    }


def test_pair_comparison_requires_exact_rollout_parameters_optimizer_and_rng():
    off = payload()
    on = copy.deepcopy(off)
    result = smoke._compare(off, on)
    assert result["changedParameterTensors"] == {"actor": 1, "critic": 1, "cost": 1}
    for field in ("rollout", "final", "optimizers", "rng"):
        bad = copy.deepcopy(on)
        if field == "rollout":
            bad[field]["digest"] = "different"
        elif field == "final":
            bad[field]["actor"]["weight"] += 1
        elif field == "optimizers":
            bad[field][0]["step"] += 1
        else:
            bad[field]["torch"] += 1
        with pytest.raises(ValueError, match=field):
            smoke._compare(off, bad)


def test_pair_ignores_only_diagnostic_producer_metadata():
    off = payload()
    on = copy.deepcopy(off)
    on["rollout"]["producerMetadataDigest"] = "economic-enabled"
    on["stats"]["worker_contracts"] = [{"economic": True}]
    assert smoke._compare(off, on)["producerMetadataDigests"]["off"] == "producer"


def test_pair_comparison_rejects_actor_or_critic_noop():
    off = payload()
    on = copy.deepcopy(off)
    for name in ("actor", "critic"):
        bad = copy.deepcopy(on)
        bad["initial"][name] = copy.deepcopy(bad["final"][name])
        off_bad = copy.deepcopy(off)
        off_bad["initial"][name] = copy.deepcopy(off_bad["final"][name])
        with pytest.raises(ValueError, match="both actor and critic"):
            smoke._compare(off_bad, bad)


def test_parser_freezes_approved_smoke_defaults():
    args = smoke._parser().parse_args(["--init", "checkpoint.pt", "--out", "output"])
    assert (args.optimizer_seed, args.world_seed, args.turns, args.lane_timeout_seconds) == (
        4242,
        50000,
        6,
        300,
    )


def test_checkpoint_personality_marks_missing_settings_as_smoke_only():
    explicit = ppo_hyper = smoke.ppo.Hyper()
    state = {"vector_critic": True, "hyper": {"personality": "dirichlet:1.0",
             "personality_scales": "reference", "personality_clip": 5.0}}
    record = smoke._checkpoint_personality(state, explicit)
    assert (ppo_hyper.personality, ppo_hyper.personality_scales, ppo_hyper.personality_clip) == (
        "dirichlet:1.0", "reference", 5.0)
    legacy = smoke.ppo.Hyper()
    record = smoke._checkpoint_personality(
        {"vector_critic": True, "hyper": {"personality": "dirichlet:1.0"}}, legacy)
    assert (legacy.personality_scales, legacy.personality_clip) == ("batch", 0.0)
    assert record["configurationSemantics"].startswith("smoke-explicit")
    assert record["initializationKind"].endswith("not-historical-resume")


@pytest.mark.parametrize("flag", ("--optimizer-seed", "--world-seed"))
def test_confirmation_seeds_are_sealed_in_parent_and_lane(flag, tmp_path, monkeypatch):
    args = ["--init", str(tmp_path / "missing.pt"), "--out", str(tmp_path / "out"), flag, "3000"]
    monkeypatch.setattr(sys, "argv", ["economic-trace-training-smoke", *args])
    with pytest.raises(Exception, match="confirmation seed denied"):
        smoke.main()
