"""Frozen TRAIN-only physical-head gate; no model calls or PPO training."""

import argparse
import base64
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from anthropocene_rl.autonomous_v5.collection import DeadlineHistoryTransport
from anthropocene_rl.autonomous_v5.contracts import observation
from anthropocene_rl.autonomous_v5.head_influence import (
    CONTROL_MAX,
    CONTROL_MIN,
    base_action,
    paired_actions,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
            raise ValueError(f"Changed frozen input: {pin['path']}")
    if len(cfg["cases"]) != 64 or cfg["decisions"] != 4:
        raise ValueError("Unexpected physical-head envelope")
    bank = Path(cfg["bank"])
    entries = {e["id"]: e for e in json.loads((bank / "manifest.json").read_text())["entries"]}
    started = time.monotonic()
    result = dict(
        version="autonomous-head-influence-v7",
        status="RUNNING",
        config_sha256=sha(args.config),
        validation_resolves=0,
        replay_resolves=0,
        model_calls=0,
        optimizer_updates=0,
        cases=[],
        control_mean_range=[CONTROL_MIN, CONTROL_MAX],
        source_bank_unchanged=True,
    )
    args.output.mkdir(parents=True)

    def save():
        result["wall_seconds"] = time.monotonic() - started
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    saved_observations = []
    try:
        command = [cfg["dotnet"], cfg["inputs"]["bridge"]["path"]]
        with DeadlineHistoryTransport(command, deadline=started + 600) as tr:
            schema = tr.schema()
            wood, clay = schema["goods"].index("Wood"), schema["goods"].index("Clay")
            for case in cfg["cases"]:
                entry = entries[case["entry"]]
                if entry["partition"] != "train":
                    raise ValueError("Head influence may use TRAIN only")
                original = (bank / entry["snapshot"]).read_bytes()
                if hashlib.sha256(original).hexdigest() != entry["sha256"]:
                    raise ValueError("Changed snapshot")
                ids = [entry["focal"]] + ([entry["peer"]] if entry["peer"] is not None else [])
                tr.reset_world(original, ids, 4)
                if case["group"] == "quotes":
                    metadata = tr._request(dict(op="autonomous5_capture", ownerDiagnostics=True))[
                        "ownerDiagnostics"
                    ]
                    schema["counterparty_ordinal"] = metadata[str(entry["focal"])][
                        "knownControlledCounterpartyOrdinals"
                    ][0]
                    schema["peer_counterparty_ordinal"] = metadata[str(entry["peer"])][
                        "knownControlledCounterpartyOrdinals"
                    ][0]
                if case.get("nonparticipating_peer"):
                    ids = ids[:1]
                snapshot = original
                if case["transform"] != "none":
                    response = tr._request(
                        dict(
                            op="autonomous5_prepare_probe",
                            snapshot=original,
                            focal=entry["focal"],
                            transform=case["transform"],
                        )
                    )
                    snapshot = base64.b64decode(response["snapshot"])
                arms = []
                for arm in range(2):
                    reset = tr.reset_world(snapshot, ids, 4)
                    current = {int(k): observation(v) for k, v in reset["observations"].items()}
                    first_obs = current[entry["focal"]]
                    if arm == 0:
                        saved_observations.append(reset["observations"][str(entry["focal"])])
                    choices = paired_actions(first_obs, case["group"], case["index"], schema)
                    trace = []
                    for decision in range(4):
                        if entry["focal"] not in current:
                            break
                        focal = entry["focal"]
                        action = choices[arm] if decision == 0 else base_action(current[focal])
                        submitted = {focal: action}
                        for peer in ids[1:]:
                            if peer not in current:
                                continue
                            p = base_action(current[peer])
                            if case["group"] == "quotes":
                                ratio = 5 if case["index"] == 1 else 0.5
                                q = max(1e-6, min(20, current[peer]["Stocks"][clay]))
                                partner = schema["peer_counterparty_ordinal"]
                                p["quote"] = dict(
                                    Counterparty=partner,
                                    Give=clay,
                                    Want=wood,
                                    GiveQuantity=q,
                                    WantQuantity=q * ratio,
                                )
                            submitted[peer] = p
                        if result["validation_resolves"] >= 512:
                            raise RuntimeError("Physical validation envelope exhausted")
                        result["validation_resolves"] += 1
                        save()
                        step = tr.step_world(submitted)["result"]
                        cap = tr._request(dict(op="autonomous5_capture", ownerDiagnostics=True))
                        trace.append(
                            dict(
                                decision=decision,
                                diagnostic=cap["ownerDiagnostics"][str(focal)],
                                reward=step["Rewards"][str(focal)],
                                fingerprint=step["Fingerprint"],
                            )
                        )
                        current = {int(k): observation(v) for k, v in step["Observations"].items()}
                    result["replay_resolves"] += len(trace)
                    if result["replay_resolves"] > 512:
                        raise RuntimeError("Physical replay envelope exhausted")
                    save()
                    tr.verify_replay()
                    arms.append(
                        dict(
                            initial_fingerprint=reset["fingerprint"],
                            requested={
                                k: np.asarray(v).tolist() if isinstance(v, np.ndarray) else v
                                for k, v in choices[arm].items()
                            },
                            trace=trace,
                            replay="PASS",
                        )
                    )
                if arms[0]["initial_fingerprint"] != arms[1]["initial_fingerprint"]:
                    raise ValueError("Paired physical worlds differ at entry")
                a, b = (arm["trace"][0]["diagnostic"] for arm in arms)
                changed = [k for k in a if isinstance(a[k], (int, float, list)) and a[k] != b[k]]
                physical = [
                    k
                    for k in changed
                    if k
                    not in {
                        "compiledStaffing",
                        "knownControlledCounterpartyOrdinals",
                        "schemaVersion",
                        "decisionDays",
                        "endDays",
                        "elapsedYears",
                    }
                ]
                result["cases"].append(
                    dict(
                        case,
                        arms=arms,
                        changed_first_receipt_fields=changed,
                        physical_changed_fields=physical,
                        different_endpoint=arms[0]["trace"][-1]["fingerprint"]
                        != arms[1]["trace"][-1]["fingerprint"],
                    )
                )
                save()
            (args.output / "observations.json").write_text(json.dumps(saved_observations) + "\n")
        result["status"] = "COMPLETE"
        result["group_summary"] = {
            g: dict(
                cases=sum(c["group"] == g for c in result["cases"]),
                first_receipt_changes=sum(
                    bool(c["changed_first_receipt_fields"])
                    for c in result["cases"]
                    if c["group"] == g
                ),
                physical_changes=sum(
                    bool(c["physical_changed_fields"]) for c in result["cases"] if c["group"] == g
                ),
            )
            for g in sorted({c["group"] for c in result["cases"]})
        }
        result["herd_growth_authority"] = "NO-GO: action3 target is opening animals times retention"
        save()
        print(json.dumps({k: v for k, v in result.items() if k != "cases"}))
    except Exception as error:
        result["status"] = "FAILED"
        result["error"] = str(error)
        save()
        raise


if __name__ == "__main__":
    main()
