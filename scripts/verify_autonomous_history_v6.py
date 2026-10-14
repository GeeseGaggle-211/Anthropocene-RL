"""Bounded TRAIN-only native history/resume/transfer proof, not a PPO campaign."""

import argparse
import base64
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from anthropocene_rl.autonomous_v4.policy import inputs as old_inputs
from anthropocene_rl.autonomous_v4.training import market_peer
from anthropocene_rl.autonomous_v5.collection import Collector, DeadlineHistoryTransport
from anthropocene_rl.autonomous_v5.contracts import HISTORY_NAMES, SLICES, observation
from anthropocene_rl.autonomous_v5.policy import inputs
from anthropocene_rl.training.autonomous_ppo_v4 import AutonomousTrainerV4
from anthropocene_rl.training.autonomous_ppo_v5 import AutonomousTrainerV5


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bridge", type=Path, required=True)
    p.add_argument("--bank", type=Path, default=Path("datasets/autonomous-ppo-gate-v1"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    start = time.monotonic()
    deadline = start + 600
    manifest = json.loads((args.bank / "manifest.json").read_text())
    entries = [e for e in manifest["entries"] if e["partition"] == "train"]
    if len(entries) != 12:
        raise ValueError("Frozen native proof expects twelve TRAIN entries")
    command = ["/workspace/Anthropocene/.tools/dotnet/dotnet", str(args.bridge.resolve())]
    result = dict(
        version="autonomous-history-native-proof-v6",
        status="RUNNING",
        validation_resolves=0,
        replay_resolves=0,
        bank_sha256=sha(args.bank / "manifest.json"),
        bridge_sha256=sha(args.bridge),
        script_sha256=sha(__file__),
        episodes=[],
        transfers=[],
        retained_native_training=0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    saved_obs = []
    try:
        with (
            DeadlineHistoryTransport(command, deadline=deadline) as tr,
            DeadlineHistoryTransport(command, deadline=deadline) as resumed,
        ):
            schema = tr.schema()
            assert (
                tuple(schema["historyFeatures"]) == HISTORY_NAMES and schema["featureWidth"] == 1061
            )
            for entry in entries:
                source = args.bank / entry["snapshot"]
                assert sha(source) == entry["sha256"]
                ids = [entry["focal"]] + ([entry["peer"]] if entry["peer"] else [])
                response = tr.reset_world(source.read_bytes(), ids, 6)
                current = {int(k): observation(v) for k, v in response["observations"].items()}
                assert all(o["History"][0] == 0 and o["History"][-2] == 1 for o in current.values())
                histories = []
                movements = trades = 0
                resumed_current = None
                for step in range(6):
                    actions = {i: market_peer(o) for i, o in current.items()}
                    focal = entry["focal"]
                    o = current[focal]
                    if (
                        entry["family"] == "trade"
                        and o["CounterpartyMask"].any()
                        and o["Stocks"][6] >= 1e-6
                    ):
                        q = max(1e-6, min(1e6, o["Stocks"][6] * 0.2))
                        actions[focal]["quote"] = dict(
                            Counterparty=int(np.flatnonzero(o["CounterpartyMask"])[0]),
                            Give=6,
                            Want=3,
                            GiveQuantity=q,
                            WantQuantity=q,
                        )
                    if step == 0 and entry["family"] == "migration" and o["MovementMask"][1:].any():
                        actions[focal]["movement"] = int(
                            np.flatnonzero(o["MovementMask"][1:])[0] + 1
                        )
                    result["validation_resolves"] += 1
                    save()
                    r = tr.step_world(actions)["result"]
                    if step == 2 and resumed_current is not None:
                        result["validation_resolves"] += 1
                        save()
                        rr = resumed.step_world(actions)["result"]
                        assert (
                            rr["Fingerprint"] == r["Fingerprint"]
                            and rr["Rewards"] == r["Rewards"]
                            and rr["CompletedHistory"] == r["CompletedHistory"]
                        )
                        result["replay_resolves"] += 1
                        save()
                        resumed.verify_replay()
                    current = {int(k): observation(v) for k, v in r["Observations"].items()}
                    for camp, history in r["CompletedHistory"].items():
                        h = np.array(history)
                        assert h[0] == 1 and h[1] == r["ElapsedYears"] and h[2] == r["ElapsedYears"]
                        assert np.array_equal(h[SLICES["compiled_staffing"]], r["Staffing"][camp])
                    movements += r["Telemetry"]["Moved"][str(focal)]
                    trades += r["Telemetry"]["SettledTrades"][str(focal)]
                    if focal in current:
                        oo = current[focal]
                        assert (oo["Context"][117:120] > 0).all()
                        histories.append(
                            dict(
                                maturity=oo["Context"][117:120].tolist(),
                                history=oo["History"].tolist(),
                                ema=oo["Context"][84:114].tolist(),
                                population=r["Telemetry"]["Population"][str(focal)],
                            )
                        )
                        if step in (0, 2, 5):
                            saved_obs.append(copy.deepcopy(oo))
                    if step == 1 and focal in current:
                        cap = tr.capture()
                        snapshot = base64.b64decode(cap["snapshot"])
                        reset = resumed.reset_world(snapshot, list(current), 4)
                        resumed_current = {
                            int(k): observation(v) for k, v in reset["observations"].items()
                        }
                        assert all(
                            np.array_equal(oo[k], resumed_current[i][k])
                            for i, oo in current.items()
                            for k in oo
                        )
                    if r["Terminated"] or r["Truncated"] or focal not in current:
                        break
                result["replay_resolves"] += step + 1
                save()
                tr.verify_replay()
                result["episodes"].append(
                    dict(
                        id=entry["id"],
                        family=entry["family"],
                        steps=step + 1,
                        history=histories,
                        moved=movements,
                        trades=trades,
                        source_sha256_unchanged=sha(source),
                        fingerprint=r["Fingerprint"],
                    )
                )
                save()
        probe = inputs(saved_obs)
        campaign = json.loads(
            Path("runs/autonomous-critic-horizon-gate-v5/summary.json").read_text()
        )
        model_for_native = None
        for cell in campaign["cells"]:
            pin = cell["health"][-1]["checkpoint"]
            path = Path(pin["path"])
            assert sha(path) == pin["sha256"]
            state = torch.load(path, map_location="cpu", weights_only=False)
            c = dict(
                state["config"],
                version="autonomous-history-v5",
                model_version="autonomous-ppo-model-v5",
                observation_version=5,
            )
            old = AutonomousTrainerV4(state["config"], 0, state["provenance"], state["width"])
            old.restore(path)
            new = AutonomousTrainerV5(
                c, 0, dict(history=5, parent_sha256=pin["sha256"]), state["width"]
            )
            new.warm_start(path, expected_sha256=pin["sha256"])
            with torch.no_grad():
                a, lp, ent, val = old.model(
                    old_inputs(saved_obs), generator=torch.Generator().manual_seed(81)
                )
                b, newlp, newent, newval = new.model(
                    probe, generator=torch.Generator().manual_seed(81)
                )
            assert torch.equal(lp, newlp) and torch.equal(ent, newent) and torch.equal(val, newval)
            assert all(
                torch.equal(getattr(a, n), getattr(b, n))
                for n in ("workforce", "activities", "controls", "movement", "trade")
            )
            assert sha(path) == pin["sha256"]
            result["transfers"].append(
                dict(
                    cell=cell["id"],
                    parent_sha256=pin["sha256"],
                    probabilities_equal=True,
                    values_equal=True,
                    observations=len(saved_obs),
                )
            )
            if cell["id"] == "small_step-h100-seed29":
                model_for_native = new
        assert model_for_native is not None
        jobs = [dict(entry=e, sampler_seed=91200 + i, limit=2) for i, e in enumerate(entries[:4])]
        collections = []
        for workers in (1, 4):
            result["validation_resolves"] += 8
            result["replay_resolves"] += 8
            save()
            with Collector(command, workers=workers, deadline=deadline) as co:
                collections.append(
                    co.collect(
                        model_for_native.model,
                        args.bank,
                        jobs,
                        model_for_native.config,
                        verify=True,
                    )
                )
        (rows, summary), (other_rows, other_summary) = collections
        assert summary == other_summary or all(
            {k: v for k, v in a.items() if k != "native_seconds"}
            == {k: v for k, v in b.items() if k != "native_seconds"}
            for a, b in zip(summary, other_summary, strict=True)
        )
        for a, b in zip(rows, other_rows, strict=True):
            assert all(torch.equal(a["obs"][k], b["obs"][k]) for k in a["obs"])
            assert (
                torch.equal(a["logp"], b["logp"])
                and a["return"] == b["return"]
                and a["advantage"] == b["advantage"]
            )
        anchors = {
            n: p.clone()
            for n, p in model_for_native.model.named_parameters()
            if not p.requires_grad
        }
        metrics = model_for_native.update(rows)
        assert all(np.isfinite(m["loss"]) for m in metrics)
        assert model_for_native.model.actor[0].history_weight.abs().sum() > 0
        assert model_for_native.model.critic[0][0].history_weight.abs().sum() > 0
        assert all(
            torch.equal(p, dict(model_for_native.model.named_parameters())[n])
            for n, p in anchors.items()
        )
        result["discarded_native_ppo_mechanical_rows"] = len(rows)
        result["discarded_ppo_updates"] = 1
        result["canonical_collection"] = "PASS"
        result["native_ppo_history_gradients"] = "PASS"
        assert result["validation_resolves"] <= 128 and result["replay_resolves"] <= 128
        result.update(
            status="PASS",
            wall_seconds=time.monotonic() - start,
            observations=len(saved_obs),
            models_converted=len(result["transfers"]),
            retained_native_training=0,
        )
        save()
        torch.save(saved_obs, args.output.with_suffix(".observations.pt"))
        print(json.dumps({k: v for k, v in result.items() if k not in ("episodes", "transfers")}))
    except BaseException as error:
        result.update(status="FAILED", error=repr(error), wall_seconds=time.monotonic() - start)
        save()
        raise


if __name__ == "__main__":
    main()
