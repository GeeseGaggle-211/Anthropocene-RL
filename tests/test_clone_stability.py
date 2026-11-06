"""Cloning stability (docs/MODEL9_TOKENS.md, "Cloning stability"): the input guard, the BC learning-rate schedule and
the phase checkpoint. No native bridge required.

The failure mode reproduced here is the one found in the stage-1 token clones: an input that is constant in the
teacher data (the movement-legal flag, 1 in every ``--model9-teacher`` row) takes another value in closed-loop
evaluation (0: ``evaluate.py`` runs with movement disabled). The 1e-3 std floor turns that into |z| = 693 on the
village token, through a weight column that never received a gradient, so the evaluated policy depends on the
random initialisation (the seed), not on what was cloned."""  # noqa: E501 - docstring/prose line kept verbatim

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from anthropocene_rl.autonomous_v7 import contracts, train
from anthropocene_rl.autonomous_v7.model import Actor, Normalizer, to_tensors

FLAG = 7  # the core column that is constant in "teacher" data


def batch(n: int = 64, seed: int = 0, core: int = 24) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    b: dict[str, np.ndarray] = {
        "core": rng.normal(size=(n, core)).astype(np.float32),
        "core_present": np.ones((n, core), bool),
        "movement_mask": np.ones((n, contracts.MOVEMENT_SLOTS), bool),
        "clothing_route_mask": np.ones((n, 2), bool),
        "shelter_route_mask": np.ones((n, 2), bool),
        "economy_mix_mask": np.ones((n, 3), bool),
    }
    b["core"][:, FLAG] = 1.0
    if core > 144:
        b["core"][:, 40] = np.arange(n) % 2
        b["core"][:, 41:53] = rng.random((n, 12))
        b["core"][:, 53] = np.arange(n) % 8
        b["core"][:, 54] = np.arange(n) % 20
        b["core"][:, 55] = 1
        b["core"][:, 144] = np.arange(n) % 2
    for name, rows, width in contracts.SETS:
        b[name] = rng.normal(size=(n, rows, width)).astype(np.float32)
        b[name + "_present"] = rng.random((n, rows)) > 0.3
    return b


def shifted(b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {k: v.copy() for k, v in b.items()}
    out["core"][:, FLAG] = 0.0  # the evaluation regime's value, never seen in training
    return out


def service_means(actor: Actor, b: dict[str, np.ndarray], normalizer: Normalizer) -> torch.Tensor:
    with torch.no_grad():
        return actor(to_tensors(b, normalizer))["service_mean"]


def test_unguarded_constant_input_shift_makes_the_policy_a_seed_lottery() -> None:
    b = batch()
    normalizer = Normalizer.fit(b)
    z = to_tensors(shifted(b), normalizer)["core"][:, FLAG]
    assert float(z.abs().min()) > 600  # symlog(1) / 1e-3
    moves = []
    for seed in range(4):
        torch.manual_seed(seed)
        actor = Actor(b["core"].shape[1], token=True).eval()
        moves.append(
            service_means(actor, shifted(b), normalizer) - service_means(actor, b, normalizer)
        )
    # The shift moves every seed's output far more than the in-distribution spread of inputs does,
    # in a direction set
    # by that seed's initialisation.
    assert all(float(m.abs().mean()) > 0.2 for m in moves)
    directions = torch.stack([m.mean(dim=0) for m in moves])
    assert float(torch.pdist(directions).min()) > 0.1


def test_guard_drops_inputs_constant_in_training_and_clips_the_rest() -> None:
    b = batch()
    normalizer = Normalizer.fit(b, guard=10.0)
    assert normalizer.dead["core"][FLAG] and normalizer.dead["core"].sum() == 1
    t = to_tensors(shifted(b), normalizer)
    assert float(t["core"][:, FLAG].abs().max()) == 0.0
    far = {k: v.copy() for k, v in b.items()}
    far["core"][:, 0] = 1e9
    assert float(to_tensors(far, normalizer)["core"][:, 0].max()) == pytest.approx(10.0)
    for seed in range(4):
        torch.manual_seed(seed)
        actor = Actor(b["core"].shape[1], token=True).eval()
        assert torch.equal(
            service_means(actor, shifted(b), normalizer), service_means(actor, b, normalizer)
        )


def test_guard_is_lossless_on_the_fitting_data() -> None:
    b = batch()
    plain, guarded = Normalizer.fit(b), Normalizer.fit(b, guard=10.0)
    a, g = to_tensors(b, plain), to_tensors(b, guarded)
    assert torch.allclose(a["core"], g["core"], atol=1e-3)  # the dead column was ~0 there already
    for name, _, _ in contracts.SETS:
        assert torch.allclose(a[name], g[name], atol=1e-3)


def test_guard_round_trips_and_unguarded_states_are_unchanged() -> None:
    b = batch()
    plain = Normalizer.fit(b).state_dict()
    assert "guard" not in plain and not any(k.endswith("_dead") for k in plain)
    assert Normalizer.from_state(plain).guard is None
    guarded = Normalizer.fit(b, guard=10.0)
    back = Normalizer.from_state(guarded.state_dict())
    assert back.guard == 10.0 and back.dead_names() == guarded.dead_names() == {"core": [FLAG]}
    assert torch.equal(
        to_tensors(shifted(b), back)["core"], to_tensors(shifted(b), guarded)["core"]
    )


def test_constant_input_receives_no_gradient() -> None:
    """Why the column stays at its random initialisation: the normalized constant is ~0 in every training row."""  # noqa: E501 - docstring/prose line kept verbatim
    b = batch()
    normalizer = Normalizer.fit(b, guard=10.0)
    torch.manual_seed(0)
    actor = Actor(b["core"].shape[1], token=True)
    actor(to_tensors(b, normalizer))["service_mean"].square().sum().backward()
    grad = actor.encoder.village.weight.grad
    assert float(grad[:, FLAG].abs().max()) == 0.0 and float(grad.abs().max()) > 0.0


def test_lr_factor_warms_up_then_cosine_decays_to_the_floor() -> None:
    total = 200
    f = [train.lr_factor(s, total, 0.05, 0.05) for s in range(total)]
    assert f[0] == pytest.approx(0.1) and f[9] == pytest.approx(1.0)
    assert all(b >= a for a, b in zip(f[:10], f[1:10], strict=False))
    assert all(b <= a + 1e-12 for a, b in zip(f[10:], f[11:], strict=False))
    assert f[-1] == pytest.approx(0.05)
    assert [train.lr_factor(s, total, 0.0, None) for s in (0, 100, 199)] == [
        1.0,
        1.0,
        1.0,
    ]  # the original flat rate


def write_shard(directory: Path, seed: int, n: int = 30, core: int = 24) -> None:
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True)
    b = batch(n, seed, core)
    arrays = dict(b)
    arrays.update(
        {
            "strategic": rng.random((n, 5)).astype(np.float32),
            "strategic_mask": np.ones((n, 5), bool),
            "service_levels": rng.random((n, len(contracts.SERVICES))).astype(np.float32),
            "service_mask": np.ones((n, len(contracts.SERVICES)), bool),
            "clothing_route": rng.integers(0, 2, n).astype(np.int32),
            "clothing_route_label_mask": np.ones(n, bool),
            "shelter_route": rng.integers(0, 2, n).astype(np.int32),
            "shelter_route_label_mask": np.ones(n, bool),
            "movement": np.zeros(n, np.int32),
            "movement_label_mask": np.ones(n, bool),
            "fission_policy": np.zeros(n, np.int32),
            "repair": np.zeros(n, np.float32),
            "camp": (np.arange(n) % 3 + 1).astype(np.int32),
            "turn": (np.arange(n) // 3).astype(np.int32),
            "society": (np.arange(n) % 3 + 1).astype(np.int32),
            "parent_society": np.zeros(n, np.int32),
            "prior_population": np.full(n, 40.0, np.float32),
            "population": np.full(n, 41.0, np.float32),
            "qol": np.full(n, 0.8, np.float32),
            "nutrition": np.ones(n, np.float32),
            "years": np.full(n, 20.0, np.float32),
            "welfare": np.full(n, 600.0, np.float32),
            "terminated": np.zeros(n, bool),
        }
    )
    for key, value in arrays.items():
        np.save(directory / f"{key}.npy", value)
    (directory / "meta.json").write_text(
        json.dumps({"seed": seed, "rows": n, "contract": contracts.NATIVE_CONTRACT})
    )


def run_main(monkeypatch, tmp_path: Path, *extra: str) -> list[dict]:
    # These historical optimizer/normalizer tests use deliberately tiny synthetic, unattributed
    # arrays.
    # Strict production sources are tested separately; keep this fixture from masquerading as
    # production evidence.
    import copy

    from test_autonomous_v7_round3_contracts import checkpoint_proof_fixture

    from anthropocene_rl.autonomous_v7 import checkpoint

    _, fixture = checkpoint_proof_fixture()
    original_load = getattr(train.data.load, "_original_load", train.data.load)

    def synthetic_load(paths, **kwargs):
        paths = list(paths)
        role = "training" if paths[0].parent.name == "train" else "selection"
        kwargs["report"].update(copy.deepcopy(fixture["family_lineage"]["teacher_inputs"][role]))
        return original_load(paths)

    synthetic_load._original_load = original_load
    monkeypatch.setattr(train.data, "load", synthetic_load)
    monkeypatch.setattr(
        checkpoint, "run_manifest", lambda *a, **kw: copy.deepcopy(fixture["run_manifest"])
    )
    out = tmp_path / "runs" / "stage1.pt"
    from anthropocene_rl.autonomous_v7.provenance import digest

    local = {
        "schema": 1,
        "execution": {"tier": "local", "purpose": "stability", "arms": {"diagnostic": [0]}},
    }
    local["digest"] = digest(local)
    local_path = tmp_path / "local.json"
    local_path.write_text(json.dumps(local))
    argv = [
        "train",
        "--train",
        str(tmp_path / "train"),
        "--dev",
        str(tmp_path / "dev"),
        "--out",
        str(out),
        "--bc-epochs",
        "2",
        "--critic-rounds",
        "1",
        "--critic-epochs",
        "1",
        "--batch",
        "16",
        "--threads",
        "1",
        "--nonqualifying",
        "--local-execution-design",
        str(local_path),
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert train.main() == 0
    return torch.load(out, map_location="cpu", weights_only=False)


def test_token_defaults_turn_the_fix_on_and_the_phase_checkpoint_resumes(
    monkeypatch, tmp_path: Path
) -> None:
    for i in range(2):
        write_shard(tmp_path / "train" / f"world-0{i}", 1000 + i)
    write_shard(tmp_path / "dev" / "world-0", 2000)
    state = run_main(monkeypatch, tmp_path, "--token")
    assert state["bc_config"] == {
        "lr": 3e-4,
        "warmup": 0.05,
        "final_lr": 0.05,
        "encoder_lr_scale": 1.0,
        "grad_clip": 1.0,
        "input_guard": 10.0,
    }
    assert state["normalizer"]["guard"] == 10.0 and state["normalizer"]["core_dead"][FLAG]
    bc = [e for e in state["log"] if "epoch" in e]
    assert len(bc) == 2 and "dev_mode" in bc[0] and bc[-1]["lr"] == pytest.approx(3e-4 * 0.05)
    phase = torch.load(tmp_path / "runs" / "stage1.bc.pt", map_location="cpu", weights_only=False)
    assert phase["normalizer"] == state["normalizer"] and phase["bc_config"] == state["bc_config"]
    # A rerun with the same --out skips cloning and keeps the phase's actor and normalizer, whatever
    # the flags say.
    again = run_main(monkeypatch, tmp_path, "--token", "--input-guard", "0")
    assert again["normalizer"] == state["normalizer"]
    assert all(torch.equal(again["actor"][k], state["actor"][k]) for k in state["actor"])


def test_pooling_defaults_reproduce_the_original_optimiser(monkeypatch, tmp_path: Path) -> None:
    for i in range(2):
        write_shard(tmp_path / "train" / f"world-0{i}", 1000 + i)
    write_shard(tmp_path / "dev" / "world-0", 2000)
    state = run_main(monkeypatch, tmp_path)
    assert state["bc_config"] == {
        "lr": 3e-4,
        "warmup": 0.0,
        "final_lr": None,
        "encoder_lr_scale": 1.0,
        "grad_clip": 1.0,
        "input_guard": None,
    }
    assert "guard" not in state["normalizer"]
    assert all(e["lr"] == pytest.approx(3e-4) for e in state["log"] if "epoch" in e)


def test_bounded_normalization_cli_persists_contract_and_rejects_legacy_resume(
    monkeypatch, tmp_path: Path
) -> None:
    for i in range(2):
        write_shard(tmp_path / "train" / f"world-0{i}", 1000 + i, core=188)
    write_shard(tmp_path / "dev" / "world-0", 2000, core=188)
    state = run_main(
        monkeypatch,
        tmp_path,
        "--token",
        "--normalization-contract",
        "bounded-v2",
        "--bc-epochs",
        "0",
        "--critic-rounds",
        "0",
    )
    assert state["feature_contract"] == "model9-bounded-physical-history-v1"
    contract = state["normalizer"]["normalization_contract"]
    assert contract["contractId"] == state["feature_contract"]
    assert contract["supportSource"] is None
    assert not any(row["supported"] for row in contract["historySupport"].values())
    phase = torch.load(tmp_path / "runs" / "stage1.bc.pt", map_location="cpu", weights_only=False)
    assert phase["feature_contract"] == state["feature_contract"]

    with pytest.raises(ValueError, match="feature contract"):
        run_main(monkeypatch, tmp_path, "--token", "--bc-epochs", "0", "--critic-rounds", "0")


def test_default_train_actor_is_bitwise_the_original_loop() -> None:
    """Pooling runs (and any run with the schedule off) reproduce: the defaults step exactly as the original code."""  # noqa: E501 - docstring/prose line kept verbatim
    b = batch(48, 3)
    rng = np.random.default_rng(5)
    n = len(b["core"])
    b.update(
        {
            "strategic": rng.random((n, 5)).astype(np.float32),
            "strategic_mask": np.ones((n, 5), bool),
            "service_levels": rng.random((n, 6)).astype(np.float32),
            "service_mask": np.ones((n, 6), bool),
            "clothing_route": np.zeros(n, np.int32),
            "clothing_route_label_mask": np.ones(n, bool),
            "shelter_route": np.ones(n, np.int32),
            "shelter_route_label_mask": np.ones(n, bool),
            "movement": np.zeros(n, np.int32),
            "movement_label_mask": np.ones(n, bool),
            "fission_policy": np.zeros(n, np.int32),
            "repair": np.zeros(n, np.float32),
            "camp": np.arange(n),
        }
    )
    normalizer = Normalizer.fit(b)
    torch.manual_seed(0)
    new = Actor(b["core"].shape[1])
    torch.manual_seed(0)
    old = Actor(b["core"].shape[1])
    train.train_actor(new, b, b, normalizer, 2, 16, 3e-4, [])
    optimizer = torch.optim.Adam(old.parameters(), lr=3e-4)  # the original loop, verbatim
    for epoch in range(2):
        old.train()
        order = np.random.default_rng(epoch).permutation(n)
        for start in range(0, n, 16):
            sub = train.subset(b, order[start : start + 16])
            loss, _ = train.bc_loss(old, to_tensors(sub, normalizer), train.label_tensors(sub))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(old.parameters(), 1.0)
            optimizer.step()
    assert all(
        torch.equal(a, c)
        for a, c in zip(new.state_dict().values(), old.state_dict().values(), strict=False)
    )
