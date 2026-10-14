"""No-learning native performance, heuristic DEV and two-learned-Elder plug checks."""

import argparse
import base64
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from anthropocene_rl.autonomous_v5.causal_budget import BudgetLedger, BudgetTransport
from anthropocene_rl.autonomous_v5.contracts import SLICES
from anthropocene_rl.autonomous_v6.contracts import validate_schema
from anthropocene_rl.policies.autonomous_inference_v6 import AutonomousElderPolicyV6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plug(cfg, ledger, output):
    bank = Path(cfg["inputs"]["bank"]["path"]).parent
    entries = json.loads((bank / "manifest.json").read_text())["entries"]
    entry = next(e for e in entries if e["id"] == cfg["development_entries"][0])
    if entry["partition"] != "development" or entry["peer"] is None:
        raise ValueError("Two-Elder plug check requires the frozen DEV pair")
    snapshot = (bank / entry["snapshot"]).read_bytes()
    if hashlib.sha256(snapshot).hexdigest() != entry["sha256"]:
        raise ValueError("Changed DEV snapshot")
    ids = [entry["focal"], entry["peer"]]
    pin = cfg["inputs"]["checkpoint"]
    state = torch.load(pin["path"], weights_only=False, map_location="cpu")
    before = sha(pin["path"])
    transport_class = type("PlugTransport", (BudgetTransport,), {"ledger": ledger})
    traces = []
    ledger.phase = "plug"
    with transport_class(
        [cfg["dotnet"], cfg["inputs"]["bridge"]["path"]], deadline=ledger.deadline
    ) as tr:
        validate_schema(tr.schema())
        for pass_ in range(2):
            reset = tr.reset_world(snapshot, ids, 4)
            current = reset["observations"]
            policies = {
                i: AutonomousElderPolicyV6(
                    pin["path"], expected_provenance=state["provenance"], seed=9900 + i
                )
                for i in ids
            }
            trace = []
            for step in range(4):
                if set(map(int, current)) != set(ids):
                    raise ValueError("Frozen plug pair did not survive all four steps")
                # Each controller receives only its own knowledge-isolated observation.
                actions = {i: policies[i].act(current[str(i)]) for i in ids}
                response = tr.step_world(actions)["result"]
                for i in ids:
                    history = response["CompletedHistory"][str(i)]
                    expected = (
                        history[SLICES["population_start"].start]
                        * history[SLICES["qol"].start]
                        * response["ElapsedYears"]
                    )
                    if not np.isclose(response["Rewards"][str(i)], expected, rtol=0, atol=1e-8):
                        raise ValueError("Village-local physical reward mismatch")
                trace.append(
                    dict(
                        fingerprint=response["Fingerprint"],
                        rewards=response["Rewards"],
                        completed_history=response["CompletedHistory"],
                        actions={
                            str(k): {
                                n: v.tolist() if isinstance(v, np.ndarray) else v
                                for n, v in a.items()
                            }
                            for k, a in actions.items()
                        },
                        observations=response["Observations"],
                    )
                )
                current = response["Observations"]
                if pass_ == 1 and step == 1:
                    tr.verify_replay()
                    captured = tr.capture()
                    rng = {i: p.generator.get_state().clone() for i, p in policies.items()}
                    resumed = tr.reset_world(base64.b64decode(captured["snapshot"]), ids, 2)
                    if resumed["observations"] != current:
                        raise ValueError("Save/load changed committed observations")
                    policies = {
                        i: AutonomousElderPolicyV6(
                            pin["path"], expected_provenance=state["provenance"], seed=0
                        )
                        for i in ids
                    }
                    for i, p in policies.items():
                        p.generator.set_state(rng[i])
                    current = resumed["observations"]
            tr.verify_replay()
            traces.append(trace)
    if traces[0] != traces[1] or before != sha(pin["path"]):
        raise ValueError("Inference/save/resume/action/reward determinism failed")
    (output / "two-elder-traces.json").write_text(json.dumps(traces, indent=2) + "\n")
    return dict(
        result="PASS",
        controlled=ids,
        native_steps=8,
        replay_steps=8,
        checkpoint_sha256=before,
        checkpoint_unchanged=True,
        saved_midway=True,
        own_observations_only=True,
        independent_rng_streams=True,
        local_reward_verified=True,
        trace_sha256=sha(output / "two-elder-traces.json"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    for pin in cfg["inputs"].values():
        if sha(pin["path"]) != pin["sha256"]:
            raise ValueError(f"Changed pinned input: {pin['path']}")
    if (
        cfg["caps"] != dict(performance=16, heuristic=50, plug=8, plug_replay=8)
        or cfg["maximum_wall_seconds"] != 1800
        or cfg["maximum_total_resolves"] != 15080
        or cfg["development_entries"]
        != ["development-872002-trade-d8", "development-872003-migration-d8"]
    ):
        raise ValueError("Frozen postflight envelope changed")
    torch.set_num_threads(1)
    prior = json.loads(Path(cfg["inputs"]["prior_budget"]["path"]).read_text())
    if prior["total_resolves"] + sum(cfg["caps"].values()) > cfg["maximum_total_resolves"]:
        raise ValueError("Postflight would exceed the global campaign")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "budget.json").exists():
        raise ValueError("Never overwrite a completed or failed run")
    ledger = BudgetLedger(
        args.output / "budget.json",
        cfg["caps"],
        prior_resolves=prior["total_resolves"],
        prior_wall_seconds=prior["cumulative_wall_seconds"],
        deadline=time.monotonic() + 1800 - prior["cumulative_wall_seconds"],
        native_version=6,
    )
    result = dict(version=cfg["version"], config_sha256=sha(args.config), status="RUNNING")
    try:
        for kind, argv in [
            ("performance", ["--autonomous-growth-benchmark"]),
            (
                "heuristic",
                [
                    "--autonomous-growth-heuristic-panel",
                    str(Path(cfg["inputs"]["bank"]["path"]).parent.resolve()),
                    *cfg["development_entries"],
                ],
            ),
        ]:
            # Conservatively charge the full attempted tool invocation before launch.
            ledger.reserve(kind, cfg["caps"][kind])
            try:
                run = subprocess.run(
                    [cfg["dotnet"], cfg["inputs"]["native_tool"]["path"], *argv],
                    text=True,
                    capture_output=True,
                    timeout=min(60, ledger.deadline - time.monotonic()),
                )
            except subprocess.TimeoutExpired as error:
                for suffix, value in [("stdout", error.stdout), ("stderr", error.stderr)]:
                    text = value.decode(errors="replace") if isinstance(value, bytes) else value
                    (args.output / f"{kind}.{suffix}").write_text(text or "")
                raise
            (args.output / f"{kind}.stdout").write_text(run.stdout)
            (args.output / f"{kind}.stderr").write_text(run.stderr)
            run.check_returncode()
            native = json.loads(run.stdout.strip().splitlines()[-1])
            if not 0 <= native["resolves"] <= cfg["caps"][kind] or (
                kind == "performance" and native["resolves"] != cfg["caps"][kind]
            ):
                raise ValueError(f"Unexpected {kind} resolve count")
            if kind == "heuristic" and native["resolves"] != sum(
                e["steps"] for e in native["episodes"]
            ):
                raise ValueError("Heuristic trace/count mismatch")
            result[kind] = native
        result["plug"] = plug(cfg, ledger, args.output)
        result["status"] = "PASS"
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        ledger.save()
        result["budget"] = json.loads(ledger.path.read_text())
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k not in ("heuristic",)}))


if __name__ == "__main__":
    main()
