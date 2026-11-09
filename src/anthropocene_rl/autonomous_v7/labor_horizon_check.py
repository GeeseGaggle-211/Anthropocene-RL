"""Horizon check CLI: paired branches to ~20 turns, per-k cumulative paired differences.

Purpose: on a small set of states, record the cumulative paired difference (a - b) at
every k = 1..k_long so a later analysis can compare how much of the long-run difference
the 2-3-turn and 5-turn + V targets recover. This module does NOT run against real
worlds: ``plan`` prints the cost estimate, ``synthetic`` runs the plumbing on the
synthetic stand-in (no evidence about the game), and ``execute`` on real worlds is
refused unless ``--i-have-approval`` is given, and even then the real-world driver is
not wired here (it raises). Worlds 3000-3059 are refused in every mode.

Cost model (matches ``search_root``'s accounting, ``2 * horizon - 1`` native resolves per
branch): per branch and stream, k labor decisions plus k - 1 shadow-H12 resolves (decision
0 reuses the root H12 labels). Per state: 2 branches x streams of each, plus one fork
and 2 x streams restores (not counted as decisions).

    python -m anthropocene_rl.autonomous_v7.labor_horizon_check plan --states 10 --k 20
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .labor_branches import paired_differences, run_paired_branches
from .labor_branches_synthetic import SYNTH_FOCAL, SyntheticNative, action_vector

SEALED_WORLDS = range(3000, 3060)
APPROVAL_FLAG = "--i-have-approval"


def cost_estimate(states: int, k: int, streams: int = 1) -> dict[str, Any]:
    """Expected simulator work for ``states`` pairs run to ``k`` decisions."""
    if states < 0 or k < 1 or streams < 1:
        raise ValueError("states >= 0, k >= 1 and streams >= 1 required")
    branches = 2 * streams
    decisions = branches * k
    shadow = branches * (k - 1)
    per_state = {
        "branches": branches,
        "labor_decisions": decisions,
        "shadow_h12_resolves": shadow,
        "native_resolves": decisions + shadow,
        "forks": 1,
        "restores": branches,
    }
    total = {key: value * states for key, value in per_state.items() if key not in ("branches",)}
    return {"states": states, "k": k, "streams": streams, "per_state": per_state, "total": total}


def format_cost(estimate: dict[str, Any]) -> str:
    p, t = estimate["per_state"], estimate["total"]
    return "\n".join(
        [
            f"states={estimate['states']} k={estimate['k']} streams={estimate['streams']}",
            f"per state: {p['branches']} branches, {p['labor_decisions']} labor decisions, "
            f"{p['shadow_h12_resolves']} shadow-H12 resolves = {p['native_resolves']} native "
            f"resolves, {p['forks']} fork, {p['restores']} restores",
            f"total:     {t['labor_decisions']} labor decisions, {t['shadow_h12_resolves']} "
            f"shadow-H12 resolves = {t['native_resolves']} native resolves, {t['forks']} forks, "
            f"{t['restores']} restores",
        ]
    )


def refuse_sealed(worlds: Sequence[int]) -> None:
    bad = sorted({int(w) for w in worlds if int(w) in SEALED_WORLDS})
    if bad:
        raise SystemExit(f"refused: worlds 3000-3059 are sealed and may not be used, got {bad[:5]}")


def load_states(path: Path) -> list[dict[str, Any]]:
    states = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            states.append(json.loads(line))
    for s in states:
        if "world" not in s:
            raise SystemExit("each state needs a world")
    return states


def run_synthetic(states: int, k: int, streams: int, seed: int) -> dict[str, Any]:
    records = []
    for i in range(states):
        native = SyntheticNative(seed=seed)
        fork = native.make_root(i)
        key = [0, seed, i, SYNTH_FOCAL]
        a = {"synthetic_vector": list(action_vector({"s": "a", "i": i}))}
        b = {"synthetic_vector": list(action_vector({"s": "b", "i": i}))}
        pair = run_paired_branches(
            native,
            root_fork=fork,
            root_h12=native.root_h12(),
            root_key=key,
            focal=SYNTH_FOCAL,
            society=1,
            founded={SYNTH_FOCAL: 0},
            first_a=a,
            first_b=b,
            k=k,
            streams=tuple(range(streams)),
        )
        diffs = paired_differences(pair)
        records.append(
            {
                "state": f"synthetic/{i}",
                "k": k,
                "streams": list(pair.stream_ids),
                "terminated": {
                    n: [bool(s["terminated"]) for s in pair.streams[n]] for n in ("a", "b")
                },
                "cumulative_paired_difference": diffs.tolist(),  # [stream][k][component]
                "decisions": pair.decisions,
            }
        )
    return {"mode": "synthetic", "k_long": k, "records": records}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("mode", choices=("plan", "synthetic", "execute"))
    p.add_argument("--states", type=int, default=10, help="number of states (plan/synthetic)")
    p.add_argument("--states-file", type=Path, help="JSON lines with world/turn/camp (execute)")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--streams", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path)
    p.add_argument("--i-have-approval", dest="approval", action="store_true")
    args = p.parse_args(argv)

    if args.states_file:
        states = load_states(args.states_file)
        refuse_sealed([s["world"] for s in states])
        count = len(states)
    else:
        states, count = [], args.states
    print(format_cost(cost_estimate(count, args.k, args.streams)))
    if args.mode == "plan":
        return 0
    if args.mode == "synthetic":
        result = run_synthetic(count, args.k, args.streams, args.seed)
        if args.out:
            args.out.write_text(json.dumps(result), encoding="utf-8")
        print(f"synthetic run complete ({len(result['records'])} states; plumbing only)")
        return 0
    if not states:
        raise SystemExit("execute needs --states-file")
    if not args.approval:
        raise SystemExit(f"refused: running on real worlds is paid compute; pass {APPROVAL_FLAG}")
    raise NotImplementedError(
        "real-world execution is deliberately not wired in this module; it needs a "
        "reviewed driver that builds the labor bridge, root forks and H12 labels"
    )


if __name__ == "__main__":
    raise SystemExit(main())
