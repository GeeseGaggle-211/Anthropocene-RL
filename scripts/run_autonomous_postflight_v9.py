"""Fixed model7 two-Elder plug, signed reversal, and explicitly cached-target audits."""

import argparse
import base64
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from train_autonomous_opportunity_v9 import pair_hash, reversal_with_adam

from anthropocene_rl.autonomous_v5.causal_budget import BudgetLedger, BudgetTransport
from anthropocene_rl.autonomous_v5.causal_diagnostics import snapshot_health
from anthropocene_rl.autonomous_v5.contracts import SLICES
from anthropocene_rl.autonomous_v6.canary_recovery import validate_probe
from anthropocene_rl.autonomous_v6.contracts import observation, validate_schema
from anthropocene_rl.autonomous_v6.policy import inputs_with_mode
from anthropocene_rl.autonomous_v6.readiness import metrics
from anthropocene_rl.policies.autonomous_inference_v7 import AutonomousElderPolicyV7
from anthropocene_rl.training.autonomous_ppo_v7 import AutonomousTrainerV7


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore(pin):
    state = torch.load(pin["path"], map_location="cpu", weights_only=False)
    trainer = AutonomousTrainerV7(state["config"], 0, state["provenance"], state["width"])
    trainer.restore(pin["path"])
    if trainer.optimizer.param_groups[0]["lr"] != 0.00003:
        raise ValueError("Postflight only covers the declared low actor LR")
    if not all(trainer.optimizer.state.get(p) for p in trainer.model.actor.parameters()):
        raise ValueError("Actor Adam must be populated")
    return trainer


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
        i: cfg["inputs"][f"checkpoint_seed{seed}"]
        for i, seed in zip(ids, [101, 211], strict=True)
    }
    states = {
        i: torch.load(pin["path"], map_location="cpu", weights_only=False)
        for i, pin in pins.items()
    }

    def policies():
        return {
            i: AutonomousElderPolicyV7(
                pins[i]["path"], expected_provenance=states[i]["provenance"], seed=9900 + i
            )
            for i in ids
        }

    transport_class = type("Model7PlugTransport", (BudgetTransport,), {"ledger": ledger})
    ledger.phase = "plug"
    traces = []
    with transport_class(
        [cfg["dotnet"], cfg["inputs"]["bridge"]["path"]], deadline=ledger.deadline
    ) as transport:
        validate_schema(transport.schema())
        for pass_ in range(2):
            current = transport.reset_world(snapshot, ids, 4)["observations"]
            controllers = policies()
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
                    rngs = {i: p.generator.get_state().clone() for i, p in controllers.items()}
                    resumed = transport.reset_world(base64.b64decode(captured["snapshot"]), ids, 2)
                    if resumed["observations"] != current:
                        raise ValueError("Save/load changed committed owner observations")
                    controllers = policies()
                    for i, p in controllers.items():
                        p.generator.set_state(rngs[i])
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
            raise ValueError("Changed prospective input: " + pin["path"])
    if (
        cfg["caps"] != dict(plug=8, plug_replay=8)
        or cfg["reversal_steps_per_phase"] != 64
        or cfg["seeds"] != [101, 211, 307]
        or cfg["maximum_total_resolves"] != 27991
        or cfg["maximum_wall_seconds"] != 2369.4072460189927
    ):
        raise ValueError("Fixed postflight envelope changed")
    prior = json.loads(Path(cfg["inputs"]["prior_budget"]["path"]).read_text())
    if prior["total_resolves"] + 16 > cfg["maximum_total_resolves"]:
        raise ValueError("Cumulative native envelope exhausted")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    adaptation = inputs_with_mode(
        [
            observation(o)
            for o in json.loads(Path(cfg["inputs"]["adaptation_probe"]["path"]).read_text())
        ]
    )
    probe_check = validate_probe(adaptation)
    args.output.mkdir(parents=True)
    ledger = BudgetLedger(
        args.output / "budget.json",
        cfg["caps"],
        prior_resolves=prior["total_resolves"],
        prior_wall_seconds=prior["cumulative_wall_seconds"],
        deadline=time.monotonic() + cfg["maximum_wall_seconds"] - prior["cumulative_wall_seconds"],
        native_version=6,
    )
    result = dict(
        version=cfg["version"], status="RUNNING", config_sha256=sha(args.config), cells=[]
    )

    def save():
        ledger.save()
        result["budget"] = json.loads(ledger.path.read_text())
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    try:
        result["probe_check"] = probe_check
        for seed in cfg["seeds"]:
            pin = cfg["inputs"][f"checkpoint_seed{seed}"]
            trainer = restore(pin)
            protected = pair_hash(trainer)
            trainer.deadline = ledger.deadline
            cached = torch.load(
                cfg["inputs"][f"cached_seed{seed}"]["path"], map_location="cpu", weights_only=False
            )
            xt, yt = cached["fitting"]
            reversal = reversal_with_adam(trainer, adaptation, steps=64, deadline=ledger.deadline)
            reversal["full_preference_switch"] = all(
                p["passed"] and p["after"][1] > 0 for p in reversal["phases"]
            )
            old_target_drift = metrics(trainer, cached["fitting"], cached["validation"])
            health = snapshot_health(trainer.model, xt, "full")
            if pair_hash(trainer) != protected or sha(pin["path"]) != pin["sha256"]:
                raise ValueError("Disposable reversal/health altered retained state")
            result["cells"].append(
                dict(
                    seed=seed,
                    checkpoint_sha256=pin["sha256"],
                    reversal=reversal,
                    cached_train_health=health,
                    cached_target_drift=old_target_drift,
                    cached_target_label=(
                        "Frozen pre-PPO actor targets; not post-PPO on-policy readiness"
                    ),
                    fitting_rows=len(yt),
                    retained_model_optimizer_rng_counters_unchanged=True,
                    protected_state_sha256=protected,
                )
            )
            save()
        result["plug"] = plug(cfg, ledger, args.output)
        result["full_reversal_pass"] = all(
            c["reversal"]["full_preference_switch"] for c in result["cells"]
        )
        result["status"] = "PASS" if result["full_reversal_pass"] else "NO-GO_REVERSAL"
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
        print(json.dumps({k: result[k] for k in ["version", "status", "budget"]}))


if __name__ == "__main__":
    main()
