"""Actual model8 two-Elder native/save/replay plug; no PPO or economic selection."""

import argparse
import base64
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from anthropocene_rl.autonomous_v5.causal_budget import BudgetLedger, BudgetTransport
from anthropocene_rl.autonomous_v5.contracts import SLICES
from anthropocene_rl.autonomous_v6.contracts import validate_schema
from anthropocene_rl.policies.autonomous_inference_v8 import AutonomousElderPolicyV8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plug(cfg, ledger, output):
    bank = Path(cfg["inputs"]["bank"]["path"]).parent
    entry = next(
        e
        for e in json.loads((bank / "manifest.json").read_text())["entries"]
        if e["id"] == cfg["plug_entry"]
    )
    if entry["partition"] != "train" or entry["world_seed"] not in cfg["fitting_worlds"]:
        raise ValueError("Plug fixture must be the pinned fitting TRAIN pair")
    ids = [entry["focal"], entry["peer"]]
    if ids[1] is None or ids[0] == ids[1]:
        raise ValueError("Two different controlled Elders required")
    snapshot = (bank / entry["snapshot"]).read_bytes()
    if hashlib.sha256(snapshot).hexdigest() != entry["sha256"]:
        raise ValueError("Changed native fixture")
    pins = {
        i: cfg["inputs"][f"checkpoint_seed{seed}"] for i, seed in zip(ids, [101, 211], strict=True)
    }
    states = {
        i: torch.load(pin["path"], map_location="cpu", weights_only=False)
        for i, pin in pins.items()
    }

    def policies():
        return {
            i: AutonomousElderPolicyV8(
                pins[i]["path"], expected_provenance=states[i]["provenance"], seed=9900 + i
            )
            for i in ids
        }

    transport_class = type("Model8PlugTransport", (BudgetTransport,), {"ledger": ledger})
    ledger.phase = "plug"
    traces = []
    with transport_class(
        [cfg["dotnet"], cfg["inputs"]["bridge"]["path"]], deadline=ledger.deadline
    ) as transport:
        validate_schema(transport.schema())
        for pass_ in range(2):
            current = transport.reset_world(snapshot, ids, 4)["observations"]
            controllers = policies()
            for i, p in controllers.items():
                p.start_episode({0: current[str(i)]})
            trace = []
            for step in range(4):
                if set(map(int, current)) != set(ids):
                    raise ValueError("Pinned pair failed to survive four common boundaries")
                # Each separate MLP/RNG sees only its owner's committed observation.
                actions = {i: controllers[i].act(current[str(i)]) for i in ids}
                result = transport.step_world(actions)["result"]
                for i in ids:
                    h = result["CompletedHistory"][str(i)]
                    expected = (
                        h[SLICES["population_start"].start]
                        * h[SLICES["qol"].start]
                        * result["ElapsedYears"]
                    )
                    if not np.isclose(result["Rewards"][str(i)], expected, rtol=0, atol=1e-8):
                        raise ValueError("Focal-only welfare identity failed")
                trace.append(
                    dict(
                        fingerprint=result["Fingerprint"],
                        rewards=result["Rewards"],
                        completed_history=result["CompletedHistory"],
                        staffing=result["Staffing"],
                        actions={
                            str(i): {
                                k: v.tolist() if isinstance(v, np.ndarray) else v
                                for k, v in a.items()
                            }
                            for i, a in actions.items()
                        },
                        observations=result["Observations"],
                    )
                )
                current = result["Observations"]
                if pass_ == 1 and step == 1:
                    transport.verify_replay()
                    captured = transport.capture()
                    sidecars = {i: p.capture_sidecar() for i, p in controllers.items()}
                    sidecar_path = output / "learner-sidecars.pt"
                    torch.save(sidecars, sidecar_path)
                    resumed = transport.reset_world(base64.b64decode(captured["snapshot"]), ids, 2)
                    if resumed["observations"] != current:
                        raise ValueError("Save/load changed committed owner observations")
                    controllers = policies()
                    restored_sidecars = torch.load(
                        sidecar_path, map_location="cpu", weights_only=False
                    )
                    for i, p in controllers.items():
                        p.restore_sidecar(restored_sidecars[i])
                    current = resumed["observations"]
            transport.verify_replay()
            traces.append(trace)
    if traces[0] != traces[1]:
        raise ValueError("Actions/rewards/staffing/observations/receipts/fingerprints differ")
    for pin in pins.values():
        if sha(pin["path"]) != pin["sha256"]:
            raise ValueError("Inference changed a retained checkpoint")
    path = output / "two-elder-traces.json"
    path.write_text(json.dumps(traces, indent=2) + "\n")
    return dict(
        status="PASS",
        controlled=ids,
        native_steps=8,
        replay_steps=8,
        checkpoints={str(i): pins[i]["sha256"] for i in ids},
        independent_models_and_rng=True,
        owner_observations_only=True,
        common_wego_boundary=True,
        exact_save_resume_and_replay=True,
        objective_p0_and_sampler_sidecar_restored=True,
        sidecar_sha256=sha(output / "learner-sidecars.pt"),
        local_reward_identity=True,
        retained_checkpoints_unchanged=True,
        trace_sha256=sha(path),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = json.loads(args.config.read_text())
    for pin in cfg["inputs"].values():
        if sha(pin["path"]) != pin["sha256"]:
            raise ValueError("Changed frozen pin: " + pin["path"])
    if cfg["caps"] != dict(plug=8, plug_replay=8):
        raise ValueError("Fixed two-Elder plug scope changed")
    prior = json.loads(Path(cfg["inputs"]["prior_budget"]["path"]).read_text())
    envelope = json.loads(Path(cfg["inputs"]["envelope"]["path"]).read_text())
    remaining = envelope["total_active_second_ceiling"] - prior["cumulative_wall_seconds"]
    if prior["total_resolves"] + 16 > envelope["total_resolve_ceiling"] or remaining <= 0:
        raise ValueError("Prospective envelope exhausted")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True)
    ledger = BudgetLedger(
        args.output / "budget.json",
        cfg["caps"],
        prior_resolves=prior["total_resolves"],
        prior_wall_seconds=prior["cumulative_wall_seconds"],
        deadline=time.monotonic() + min(remaining, 60),
        native_version=6,
    )
    result = dict(version=cfg["version"], status="RUNNING", config_sha256=sha(args.config))
    try:
        result["plug"] = plug(cfg, ledger, args.output)
        result["status"] = "PASS"
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        ledger.save()
        result["budget"] = json.loads(ledger.path.read_text())
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: result[k] for k in ["status", "budget"]}))


if __name__ == "__main__":
    main()
