"""Scripted physical event witnesses, deliberately separate from PPO evidence."""

import argparse
import json
from pathlib import Path

import numpy as np

from anthropocene_rl.autonomous_v4.contracts import observation
from anthropocene_rl.autonomous_v4.env import AutonomousTransport
from anthropocene_rl.autonomous_v4.training import market_peer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries = json.loads((args.bank / "manifest.json").read_text())["entries"][:4]
    results = []
    with AutonomousTransport(
        ["/workspace/Anthropocene/.tools/dotnet/dotnet", str(args.bridge.resolve())]
    ) as transport:
        for entry in entries:
            focal = entry["focal"]
            ids = [focal] + ([entry["peer"]] if entry["peer"] else [])
            response = transport.reset_world((args.bank / entry["snapshot"]).read_bytes(), ids, 8)
            current = {int(k): observation(v) for k, v in response["observations"].items()}
            totals = dict(moved=0, trades=0, fissions=0, migrants=0)
            for step in range(8):
                actions = {i: market_peer(o) for i, o in current.items()}
                obs = current[focal]
                if (
                    entry["family"] == "trade"
                    and obs["CounterpartyMask"].any()
                    and obs["Stocks"][6] >= 1e-6
                ):
                    quantity = max(1e-6, min(1e6, obs["Stocks"][6] * 0.2))
                    actions[focal]["quote"] = dict(
                        Counterparty=int(np.flatnonzero(obs["CounterpartyMask"])[0]),
                        Give=6,
                        Want=3,
                        GiveQuantity=quantity,
                        WantQuantity=quantity,
                    )
                if step == 0 and obs["MovementMask"][1:].any():
                    actions[focal]["movement"] = int(np.flatnonzero(obs["MovementMask"][1:])[0] + 1)
                result = transport.step_world(actions)["result"]
                t = result["Telemetry"]
                totals["moved"] += t["Moved"][str(focal)]
                totals["trades"] += t["SettledTrades"][str(focal)]
                totals["fissions"] += t["DaughterSettlements"][str(focal)]
                totals["migrants"] += t["Immigrated"][str(focal)] + t["Emigrated"][str(focal)]
                current = {int(k): observation(v) for k, v in result["Observations"].items()}
                if focal not in current or result["Truncated"]:
                    break
            transport.verify_replay()
            results.append(dict(id=entry["id"], events=totals, fingerprint=result["Fingerprint"]))
    total = {k: sum(r["events"][k] for r in results) for k in totals}
    if not all(total.values()):
        raise ValueError(f"Missing event witness: {total}")
    args.output.write_text(
        json.dumps(
            dict(kind="scripted-not-learned", episodes=results, total=total, replay="PASS"),
            indent=2,
        )
    )
    print(json.dumps(total))


if __name__ == "__main__":
    main()
