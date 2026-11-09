"""Reduce movement_audit rollouts to the tables of docs/MODEL9_MOVEMENT_AUDIT.md.

    python -m anthropocene_rl.autonomous_v7.movement_audit_summary --runs dir --tags s55-grouped,s55-argmax,ref-native,s55-native --out summary.json

Physical move: a camp's tile differs between consecutive inspects. Directive: a learned keep-intent (-3) or a region row
other than the current tile (101-107). Tenure: consecutive turns at the tile, counting the turn of arrival as 1.
"""  # noqa: E501 - docstring/prose line kept verbatim

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

KEEP, REGION_BASE = -3, 100


def is_directive(code: int) -> bool:
    return code == KEEP or REGION_BASE < code < REGION_BASE + 8


def m(xs: list[float]) -> float:
    return float(mean(xs)) if xs else float("nan")


def food_split(h: dict[str, float]) -> dict[str, float]:
    return {
        "land": h["terrestrial"],
        "fish": h["marine"] + h["lake"] + h["fish"],
        "herd": h["animal"] + h["hunt"],
        "total": h["produced"],
    }


def world(path: Path) -> dict[str, Any]:
    d = json.loads(path.read_text())
    rows = d["rows"]
    out: dict[str, Any] = defaultdict(list)
    tenure: dict[int, int] = {}
    episode: dict[int, tuple[int, int]] = {}  # camp -> (start turn index, target tile)
    lineage: dict[int, int] = {}
    welfare = welfare_e1 = 0.0
    traj: dict[int, list[dict[str, Any]]] = defaultdict(list)  # camp -> per-turn post states
    abandoned_at: dict[int, int] = {}
    visited: dict[int, list[int]] = {}
    for c in rows[0]["pre"]["camps"]:
        tenure[c["camp"]] = (
            3  # openings: settled at least a few turns (heuristic opening); treated as incumbent
        )
    for i, row in enumerate(rows[:-1]):
        pre = {c["camp"]: c for c in row["pre"]["camps"]}
        post_all = rows[i + 1]["pre"]
        post = {c["camp"]: c for c in post_all["camps"]}
        directives = {int(k): int(v) for k, v in row["directives"].items()}
        for v in row["villages"]:
            soc, parent = int(v["society"]), int(v.get("parentSociety", 0) or 0)
            if soc not in lineage:
                lineage[soc] = lineage.get(parent, soc) if parent else soc
        for v in row["villages"]:
            if v["priorPopulation"] <= 0:
                continue
            welfare += v["welfare"]
            c = post.get(v["camp"])
            factor = c["gearSettled"] / c["gear"] if c and c["gear"] > 1e-9 else 1.0
            welfare_e1 += v["welfare"] * factor
        for camp, c in post.items():
            traj[camp].append(
                {"i": i, **{k: c[k] for k in ("qol", "population", "gear", "gearSettled", "tile")}}
            )
        for camp, b in pre.items():
            a = post.get(camp)
            code = directives.get(camp)
            if code is not None:
                out["decisions"].append(1)
                out["directive"].append(int(is_directive(code)))
            if a is None:
                continue
            moved = a["tile"] != b["tile"]
            settled = b["isVillage"] or b["fixedHere"] > 1e-6 or b["fieldTiles"] > 0
            out["camp_turns"].append(1)
            out["moved"].append(int(moved))
            out["moved_settled" if settled else "moved_nomadic"].append(int(moved))
            # Relocation episodes (learned policies): from the first directive to arrival at its
            # target.
            if code is not None and is_directive(code):
                target = a["intentTile"] if a["hasIntent"] else (a["tile"] if moved else -1)
                if camp not in episode or (
                    code != KEEP
                    and episode[camp][1] != target
                    and not (not a["hasIntent"] and moved)
                ):
                    episode[camp] = (i, target)
                if moved and not a["hasIntent"]:
                    out["relocation_turns"].append(i - episode[camp][0] + 1)
                    episode.pop(camp, None)
                elif target == -1:
                    out["relocation_cancelled"].append(1)
                    episode.pop(camp, None)
            elif camp in episode:
                out["relocation_cancelled"].append(1)
                episode.pop(camp)
            if moved:
                neighbours = {n["tile"]: n for n in b["neighbours"]}
                out["move_adjacent"].append(int(a["tile"] in neighbours))
                left = next((f["units"] for f in a["fixedOwned"] if f["tile"] == b["tile"]), 0.0)
                waiting = next((f["units"] for f in b["fixedOwned"] if f["tile"] == a["tile"]), 0.0)
                history = visited.setdefault(camp, [])
                out["return_2"].append(int(a["tile"] in history[-2:]))
                out["return_any"].append(int(a["tile"] in history))
                out["capital"].append(
                    {
                        "settled": settled,
                        "fixed_before": b["fixedHere"],
                        "fixed_left": left,
                        "fixed_after": a["fixedHere"],
                        "fixed_waiting": waiting,
                        "field_tiles": b["fieldTiles"],
                        "field_capital": b["fieldCapital"],
                        "local_capital": b["localCapital"],
                        "stores_before": b["stores"],
                        "stores_after": a["stores"],
                        "light": b["lightUnits"],
                        "light_after": a["lightUnits"],
                        "village_before": b["isVillage"],
                        "village_after": a["isVillage"],
                        "pre_food": b["food"],
                        "pre_e": b["expectation"],
                        "post_e": a["expectation"],
                        "pre_pop": b["population"],
                        "gear_gain": a["gear"] / max(1e-9, a["gearSettled"]) - 1,
                    }
                )
                arrival = neighbours.get(a["tile"])
                if arrival is not None and arrival["landCap"] > 0:
                    out["arrival_stock"].append(arrival["land"] / arrival["landCap"])
                    out["origin_stock"].append(
                        b["footprint"]["land"] / max(1e-9, b["footprint"]["landCap"])
                    )
                    out["arrival_vs_origin_land"].append(
                        arrival["land"] / max(1e-9, b["footprint"]["land"])
                    )
                abandoned_at[b["tile"]] = i
                out["coast_before"].append(int(b["footprint"]["shallows"] > 0))
                out["coast_after"].append(int(a["footprint"]["shallows"] > 0))
                tenure[camp] = 1
            else:
                tenure[camp] = tenure.get(camp, 1) + 1
            visited.setdefault(camp, [])
            if not visited[camp] or visited[camp][-1] != b["tile"]:
                visited[camp].append(b["tile"])
            if b["footprint"]["landCap"] > 0 and not moved and tenure[camp] >= 4:
                out["steady_stock"].append(
                    b["footprint"]["land"] / b["footprint"]["landCap"]
                )  # >= 3 turns of occupation
            h = a.get("harvest")
            if h and h["start"] > 0:
                f = food_split(h)
                key = "fresh" if moved else ("steady" if tenure[camp] >= 3 else "middle")
                f.update(
                    births=h["births"],
                    deaths=h["deaths"],
                    required=h["required"],
                    consumed=h["consumed"],
                )
                out["harvest_" + key].append({k: x / h["start"] for k, x in f.items()})
                out["lifestyle_" + key].append(int(settled))
        for camp, a in post.items():  # noqa: B007 - loop variable name kept (renaming skipped)
            if camp not in pre:
                tenure[camp] = 1
        # Regrowth of abandoned footprints (queried each turn after abandonment).
        for t in row["post_tiles"]:
            if t["tile"] in abandoned_at and t["landCap"] > 0:
                out["regrowth"].append((i - abandoned_at[t["tile"]], t["land"] / t["landCap"]))
        for f in post_all.get("fissions") or []:
            parent = f["parent"]
            moved_recently = (
                parent in pre and parent in post and pre[parent]["tile"] != post[parent]["tile"]
            )
            out["fissions"].append(
                {
                    "lineage": lineage.get(f["parentSociety"], f["parentSociety"]),
                    "moved": moved_recently,
                    "trigger": f["trigger"],
                    "people": f["people"],
                    "onto_old_tile": int(moved_recently and f["tile"] == pre[parent]["tile"]),
                    "near_old_tile": int(
                        moved_recently
                        and any(n["tile"] == f["tile"] for n in pre[parent]["neighbours"])
                    ),
                }
            )
        out["migrations"].append(post_all.get("migrations", 0))
    # After arrival: physical moves at index i versus same-turn incumbents (tenure >= 3, did not
    # move at i).
    final = rows[-1]["pre"]["camps"]
    people = sum(c["population"] for c in final)
    aquatic = sum(
        food_split(c["harvest"])["fish"]
        for r in rows[1:]
        for c in r["pre"]["camps"]
        if c.get("harvest")
    )
    total = sum(
        c["harvest"]["produced"] for r in rows[1:] for c in r["pre"]["camps"] if c.get("harvest")
    )
    return {
        "seed": d["seed"],
        "policy": d["policy"],
        "out": dict(out),
        "traj": dict(traj),
        "welfare": welfare,
        "welfare_e1": welfare_e1,
        "residents": people,
        "settlements": len(final),
        "below_30": sum(1 for c in final if c["population"] < 30),
        "qol": sum(c["qol"] * c["population"] for c in final) / max(1, people),
        "qol_e1": sum(
            c["qol"] * c["gearSettled"] / max(1e-9, c["gear"]) * c["population"] for c in final
        )
        / max(1, people),
        "expectation": sum(c["expectation"] * c["population"] for c in final) / max(1, people),
        "aquatic_share": aquatic / max(1e-9, total),
        "coastal_share": sum(c["population"] for c in final if c["footprint"]["shallows"] > 0)
        / max(1, people),
        "villages": sum(1 for c in final if c["isVillage"]),
        "fixed_units": sum(c["fixedHere"] for c in final),
        "lineages": len({lineage.get(c["society"], c["society"]) for c in final}),
    }


def after_arrival(worlds: list[dict[str, Any]]) -> dict[str, Any]:
    """QoL, expectation-neutral QoL (E = 1) and population at arrival (k=0) and k=1..3 turns later, relative to the turn
    before the move. Movers are compared with matched non-movers: same world, same turn, same log2 population bucket,
    settled at their tile for the two turns before. Reported: mover mean, matched-control mean and their difference."""  # noqa: E501 - docstring/prose line kept verbatim
    import math

    events: list[tuple[str, int, int, int, list[tuple[float, float, float]]]] = []
    for w in worlds:
        for _camp, t in w["traj"].items():
            by = {s["i"]: s for s in t}
            for s in t:
                i = s["i"]
                prev = by.get(i - 1)
                if prev is None or prev["population"] <= 0:
                    continue
                moved = s["tile"] != prev["tile"]
                if not moved and any(
                    by.get(i - j) is None or by[i - j]["tile"] != s["tile"] for j in (1, 2)
                ):
                    continue
                e1p = prev["qol"] * prev["gearSettled"] / max(1e-9, prev["gear"])
                path = []
                for k in range(4):
                    x = by.get(i + k)
                    if x is None or x["population"] <= 0:
                        break
                    e1 = x["qol"] * x["gearSettled"] / max(1e-9, x["gear"])
                    path.append(
                        (x["qol"] - prev["qol"], e1 - e1p, x["population"] / prev["population"] - 1)
                    )
                bucket = int(math.log2(max(1, prev["population"])))
                events.append(("m" if moved else "s", w["seed"], i, bucket, path))
    controls: dict[tuple[int, int, int, int], list[tuple[float, float, float]]] = defaultdict(list)
    for kind, seed, i, b, path in events:
        if kind == "s":
            for k, x in enumerate(path):
                controls[(seed, i, b, k)].append(x)
    out: dict[int, dict[str, Any]] = {}
    for k in range(4):
        mv, ct = [], []
        for kind, seed, i, b, path in events:
            if kind != "m" or len(path) <= k or not controls.get((seed, i, b, k)):
                continue
            c = controls[(seed, i, b, k)]
            mv.append(path[k])
            ct.append(tuple(m([x[j] for x in c]) for j in range(3)))
        out[k] = {
            "n": len(mv),
            **{
                name + suffix: m([x[j] for x in src])
                for j, name in enumerate(("dqol", "dqol_e1", "dpop"))
                for suffix, src in (("_mover", mv), ("_control", ct))
            },
        }
    return out


def reduce(worlds: list[dict[str, Any]]) -> dict[str, Any]:
    pool: dict[str, list[Any]] = defaultdict(list)
    for w in worlds:
        for k, v in w["out"].items():
            pool[k] += v

    def share(key: str) -> float:
        return m(pool[key])

    def harvest(key: str) -> dict[str, float]:
        rows = pool["harvest_" + key]
        return {
            "n": len(rows),
            **{
                k: m([r[k] for r in rows])
                for k in (
                    "land",
                    "fish",
                    "herd",
                    "total",
                    "births",
                    "deaths",
                    "required",
                    "consumed",
                )
            },
            "settled_share": m(pool["lifestyle_" + key]),
        }

    capital = pool["capital"]
    regrowth: dict[int, list[float]] = defaultdict(list)
    for age, ratio in pool["regrowth"]:
        regrowth[age].append(ratio)
    lineage_fissions: dict[tuple[int, int], int] = defaultdict(int)
    for w in worlds:
        for f in w["out"].get("fissions", []):
            lineage_fissions[(w["seed"], f["lineage"])] += 1
    fissions = pool["fissions"]
    per_world = lambda k: m([w[k] for w in worlds])  # noqa: E731 - lambda assignment kept verbatim
    return {
        "worlds": len(worlds),
        "welfare": sum(w["welfare"] for w in worlds),
        "welfare_e1": sum(w["welfare_e1"] for w in worlds),
        "residents": per_world("residents"),
        "settlements": per_world("settlements"),
        "below_30": per_world("below_30"),
        "below_30_share": sum(w["below_30"] for w in worlds)
        / max(1, sum(w["settlements"] for w in worlds)),
        "qol": sum(w["qol"] * w["residents"] for w in worlds)
        / max(1, sum(w["residents"] for w in worlds)),
        "qol_e1": sum(w["qol_e1"] * w["residents"] for w in worlds)
        / max(1, sum(w["residents"] for w in worlds)),
        "expectation": sum(w["expectation"] * w["residents"] for w in worlds)
        / max(1, sum(w["residents"] for w in worlds)),
        "villages": per_world("villages"),
        "fixed_units": per_world("fixed_units"),
        "lineages": per_world("lineages"),
        "directive_share": share("directive") if pool["directive"] else None,
        "physical_move_share": share("moved"),
        "move_share_settled": share("moved_settled"),
        "move_share_nomadic": share("moved_nomadic"),
        "settled_camp_turn_share": len(pool["moved_settled"]) / max(1, len(pool["moved"])),
        "moves_adjacent": share("move_adjacent"),
        "return_within_2": share("return_2"),
        "return_ever": share("return_any"),
        "moves": len(pool["moved"]) and sum(pool["moved"]),
        "relocation_turns_mean": m(pool["relocation_turns"]),
        "relocation_turns_median": float(median(pool["relocation_turns"]))
        if pool["relocation_turns"]
        else None,
        "relocations_completed": len(pool["relocation_turns"]),
        "relocations_cancelled": len(pool["relocation_cancelled"]),
        "capital_per_move": {
            "n": len(capital),
            "settled_share": m([c["settled"] for c in capital]),
            "fixed_before": m([c["fixed_before"] for c in capital]),
            "fixed_left": m([c["fixed_left"] for c in capital]),
            "fixed_after": m([c["fixed_after"] for c in capital]),
            "fixed_waiting": m([c["fixed_waiting"] for c in capital]),
            "dest_has_own_fixed": m([int(c["fixed_waiting"] > 1e-6) for c in capital]),
            "field_tiles": m([c["field_tiles"] for c in capital]),
            "field_capital": m([c["field_capital"] for c in capital]),
            "local_capital": m([c["local_capital"] for c in capital]),
            "stores_before": m([c["stores_before"] for c in capital]),
            "stores_after": m([c["stores_after"] for c in capital]),
            "light_before": m([c["light"] for c in capital]),
            "light_after": m([c["light_after"] for c in capital]),
            "village_lost": m(
                [int(c["village_before"] and not c["village_after"]) for c in capital]
            ),
            "village_before": m([int(c["village_before"]) for c in capital]),
            "pre_food": m([c["pre_food"] for c in capital]),
            "pre_e": m([c["pre_e"] for c in capital]),
            "post_e": m([c["post_e"] for c in capital]),
            "pre_pop": m([c["pre_pop"] for c in capital]),
            "gear_gain_median": float(median([c["gear_gain"] for c in capital]))
            if capital
            else None,
            "with_fixed": m([int(c["fixed_before"] > 1e-6) for c in capital]),
            "with_fields": m([int(c["field_tiles"] > 0) for c in capital]),
        },
        "harvest_fresh": harvest("fresh"),
        "harvest_steady": harvest("steady"),
        "harvest_middle": harvest("middle"),
        "arrival_stock_ratio": m(pool["arrival_stock"]),
        "origin_stock_ratio": m(pool["origin_stock"]),
        "arrival_vs_origin_land": float(median(pool["arrival_vs_origin_land"]))
        if pool["arrival_vs_origin_land"]
        else None,
        "steady_stock_ratio": m(pool["steady_stock"]),
        "regrowth": {
            age: {"n": len(v), "ratio": m(v)} for age, v in sorted(regrowth.items()) if age <= 6
        },
        "fissions_per_world": len(fissions) / max(1, len(worlds)),
        "fissions_after_move_share": m([int(f["moved"]) for f in fissions]),
        "fission_daughter_on_parent_old_tile": m(
            [f["onto_old_tile"] for f in fissions if f["moved"]]
        ),
        "fission_daughter_next_to_old_tile": m(
            [f["near_old_tile"] for f in fissions if f["moved"]]
        ),
        "fission_people_mean": m([f["people"] for f in fissions]),
        "fissions_per_lineage": m(list(lineage_fissions.values())) if lineage_fissions else 0.0,
        "fission_triggers": {
            t: sum(1 for f in fissions if f["trigger"] == t)
            for t in sorted({f["trigger"] for f in fissions})
        },
        "migrations_per_world": sum(sum(w["out"].get("migrations", [])) for w in worlds)
        / max(1, len(worlds)),
        "coast_before_move": share("coast_before"),
        "coast_after_move": share("coast_after"),
        "aquatic_share": per_world("aquatic_share"),
        "coastal_share": per_world("coastal_share"),
        "after_arrival": after_arrival(worlds),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--tags", required=True)
    parser.add_argument(
        "--fishing-share",
        type=float,
        default=0.25,
        help="fishing world: aquatic share of the first tag's food at least this",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tags = args.tags.split(",")
    data = {
        t: sorted((world(p) for p in args.runs.glob(f"{t}-*.json")), key=lambda w: w["seed"])
        for t in tags
    }
    seeds = sorted(set.intersection(*[{w["seed"] for w in ws} for ws in data.values()]))
    data = {t: [w for w in ws if w["seed"] in seeds] for t, ws in data.items()}
    base = {w["seed"]: w for w in data[tags[-1]]}
    fishing = sorted(s for s, w in base.items() if w["aquatic_share"] >= args.fishing_share)
    report: dict[str, Any] = {
        "seeds": seeds,
        "fishing_seeds": fishing,
        "fishing_reference": tags[-1],
        "policies": {},
    }
    for t, ws in data.items():
        report["policies"][t] = {
            "all": reduce(ws),
            "fishing": reduce([w for w in ws if w["seed"] in fishing]),
            "per_seed": [
                {
                    k: w[k]
                    for k in (
                        "seed",
                        "welfare",
                        "welfare_e1",
                        "residents",
                        "settlements",
                        "qol",
                        "qol_e1",
                        "aquatic_share",
                        "coastal_share",
                    )
                }
                for w in ws
            ],
        }
    args.out.write_text(json.dumps(report, indent=1, default=str))
    for t in tags:
        a = report["policies"][t]["all"]
        print(
            t,
            json.dumps(
                {
                    k: a[k]
                    for k in (
                        "welfare",
                        "welfare_e1",
                        "residents",
                        "settlements",
                        "qol",
                        "qol_e1",
                        "directive_share",
                        "physical_move_share",
                        "relocation_turns_mean",
                    )
                }
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
