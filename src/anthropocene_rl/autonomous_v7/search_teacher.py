"""Search teacher (expert iteration, docs/MODEL9_SEARCH_TEACHER.md): improves on a policy by simulation and writes the
winning alternatives as imitation labels.

At sampled decision points the bridge forks the world (op fork; every restore replays the live path exactly). For one
village it rolls out each alternative for ``horizon`` turns: the policy itself, the native heuristic, each legal region
move (then keep the intent) and each pure economy mix (held for the horizon). Every other village follows the
policy deterministically, so branches differ only by the alternative (paired comparisons with no world noise). The
score is the village lineage's discounted welfare over the horizon plus the critic's value of the lineage at the end
(self-interested credit: the lineage's own settlements, founded daughters included). The label row is the village's
observation at the fork, the best alternative, and every alternative's score; PPO imitates rows whose gain clears a
margin (``ppo --search-labels``).

    python -m anthropocene_rl.autonomous_v7.search_teacher --checkpoint <pt> --seeds 1000-1011 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import contracts, data
from .bridge import Model9Bridge, observation_batch
from .evaluate import load, policy_actions
from .model import Critic, VectorCritic, mix_shares, to_tensors
from .ppo import head_scale, personality_sampler, village_batch
from .upgrade import upgrade

# Alternative kinds (label_kind): the policy's own action, the heuristic, a region move, a pure economy mix.
POLICY, NATIVE, REGION, MIX = 0, 1, 2, 3
KINDS = ("policy", "native", "region", "mix")
MAX_ALTERNATIVES = 2 + contracts.REGION_ROWS - 1 + len(contracts.ECONOMY_MIX)


@dataclass
class SearchConfig:
    turns: int = 25             # episode length of the main path
    horizon: int = 6            # turns each alternative is rolled out (120 years) before the critic bootstrap
    points: int = 4             # decision turns searched per world
    villages: int = 2           # villages searched per decision turn
    first_turn: int = 1
    regions: bool = True
    mixes: bool = True
    native: bool = True
    # Add the critic's end-of-horizon lineage value to the score. Off by default: the first labels (20 October 2026)
    # showed the critic, trained on Elder-acting states, scores native and moved states off-distribution (native
    # branches lost 15% on the bootstrap at equal returns). The bootstrap is always recorded as a diagnostic.
    bootstrap: bool = False
    # Personalities (docs/MODEL9_PERSONALITY.md): each society is drawn from this distribution ("default", "fixed:..."
    # or "dirichlet:a", as in ppo --personality) and daughters inherit their parent's draw. Every alternative returns
    # its lineage's discounted return per objective (contracts.OBJECTIVES), and a decision point's label is scored for
    # its society's personality: 1 + sum_k p_k (R_k(alt) - R_k(policy)) / |R_k(policy)|. The policy, the critic and
    # the imitation loss all read that personality, through the same FiLM-conditioned encoder.
    personality: str = "default"


def load_critic(checkpoint: Path):
    state = upgrade(torch.load(checkpoint, map_location="cpu", weights_only=False))
    core, privileged, token = int(state["core_width"]), int(state.get("privileged_width", 0)), bool(state.get("token", False))
    critic = (VectorCritic if state.get("vector_critic") else Critic)(core, privileged_width=privileged, token=token)
    critic.load_state_dict(state["critic"]); critic.eval()
    return critic


def lineage_values(critic, normalizer, villages: list[dict[str, Any]], lineage: set[int],
                   personalities: dict[int, Any] | None = None) -> np.ndarray:
    """The critic's value of the lineage's living settlements per objective (contracts.OBJECTIVES; a scalar critic
    fills welfare only, the rest NaN), each head unscaled as ppo.head_scale says, at each settlement's personality."""
    out_values = np.full(len(contracts.OBJECTIVES), np.nan); out_values[0] = 0.0
    rows = [v for v in villages if v.get("observation") is not None and not v.get("terminated") and v["society"] in lineage]
    if not rows:
        return np.zeros(len(contracts.OBJECTIVES))  # no living settlement: nothing left to value
    batch = village_batch(rows)
    if personalities:
        default = [contracts.OBJECTIVE_WEIGHTS[k] for k in contracts.OBJECTIVES]
        batch["personality"] = np.asarray([personalities.get(int(v["society"]), default) for v in rows], np.float32)
    population = np.asarray([v["population"] for v in rows], dtype=np.float64)
    with torch.no_grad():
        out = critic(to_tensors(batch, normalizer))
    heads = out if isinstance(out, dict) else {"welfare": out}
    for k, name in enumerate(contracts.OBJECTIVES):
        if name in heads:
            u = critic.expectation(heads[name]).numpy().astype(np.float64)
            out_values[k] = float((u * head_scale(name, population)).sum())
    return out_values


def lineage_value(critic, normalizer, villages: list[dict[str, Any]], lineage: set[int]) -> float:
    """The critic's welfare value of the lineage's living settlements: u x residents, summed."""
    return float(lineage_values(critic, normalizer, villages, lineage)[0])


def component_rewards(villages: list[dict[str, Any]], lineage: set[int]) -> np.ndarray:
    """This step's lineage reward per objective (contracts.OBJECTIVES order): welfare P x Q x dt and
    population-years P x dt summed over living settlements, realised trade utility, and QoL-years as the lineage's
    population-weighted QoL x dt. That last one is one number per lineage, so splitting into more settlements is not
    rewarded as more QoL."""
    out = np.zeros(len(contracts.OBJECTIVES))
    people = weighted = years = 0.0
    for v in villages:
        if v["society"] not in lineage:
            continue
        alive = not v["terminated"] and v["priorPopulation"] > 0
        dt = float(v.get("years", 0.0))
        out[0] += float(v["welfare"])
        out[2] += float(v.get("tradeUtility", 0.0))
        if alive:
            out[1] += float(v["priorPopulation"]) * dt
            people += float(v["priorPopulation"]); weighted += float(v["priorPopulation"]) * float(v["qol"]); years = dt
    out[3] = weighted / people * years if people > 0 else 0.0
    return out


def personality_score(returns: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Scores of a decision point's alternatives (rows of ``returns``, the policy first) for personality ``p``:
    1 + sum_k p_k (R_k - R_k(policy)) / |R_k(policy)|. The policy scores 1; at the default personality this is the
    welfare ratio to the policy."""
    base = returns[0]
    scale = np.where(np.abs(base) > 1e-9, np.abs(base), 1.0)
    weights = np.where(np.isnan(returns).any(axis=0), 0.0, p)
    return 1.0 + (np.nan_to_num(returns - base) / scale * weights).sum(axis=1)


def alternatives(village: dict[str, Any], base: dict[str, Any], config: SearchConfig) -> list[dict[str, Any]]:
    """The alternatives searched for one village at a fork, the policy first."""
    obs = village["observation"]
    out = [{"kind": POLICY, "slot": -1, "mix": None}]
    if config.native:
        out.append({"kind": NATIVE, "slot": contracts.REGION_NATIVE, "mix": None})
    if config.regions:
        mask = obs.get("regionMovementMask") or []
        for row in range(1, contracts.REGION_ROWS):  # row 0 is the current tile (stay)
            slot = 2 + row
            if slot < len(mask) and mask[slot]:
                out.append({"kind": REGION, "slot": slot, "mix": None})
    if config.mixes:
        legal = obs["economyMixMask"]
        s, f = base["strategic"][3], base["strategic"][4]
        policy_mix = np.asarray([(1 - f) * s, (1 - f) * (1 - s), f])
        for j, ok in enumerate(legal):
            if ok and sum(legal) > 1:
                vertex = np.eye(len(contracts.ECONOMY_MIX))[j]
                if 0.5 * np.abs(vertex - policy_mix).sum() > 0.05:
                    out.append({"kind": MIX, "slot": -1, "mix": vertex})
    return out


def override(actions: dict[int, dict[str, Any]], camp: int, alt: dict[str, Any], village: dict[str, Any] | None,
             first: bool) -> dict[int, dict[str, Any]]:
    """This turn's actions with the alternative applied to ``camp`` (the policy's elsewhere)."""
    actions = dict(actions)
    if alt["kind"] == POLICY or camp not in actions:
        return actions
    if alt["kind"] == NATIVE:
        actions.pop(camp)
        return actions
    action = dict(actions[camp])
    if alt["kind"] == REGION:
        mask = (village or {}).get("observation", {}).get("regionMovementMask") or []
        if first:
            action["movement"] = contracts.REGION_BASE + alt["slot"] - 2
        elif len(mask) > contracts.REGION_KEEP and mask[contracts.REGION_KEEP]:
            action["movement"] = contracts.KEEP_INTENT
    elif alt["kind"] == MIX:
        s, f = mix_shares(alt["mix"][None, :])
        strategic = list(action["strategic"]); strategic[3] = float(s[0]); strategic[4] = float(f[0])
        action["strategic"] = strategic
    actions[camp] = action
    return actions


def inherit(personalities: dict[int, Any] | None, villages: list[dict[str, Any]]) -> None:
    """Daughters founded in a branch take their parent's personality exactly, so every branch of a fork scores under
    the same personalities (training adds drift; search keeps branches paired)."""
    if personalities is None:
        return
    for v in villages:
        society, parent = int(v["society"]), int(v.get("parentSociety", 0))
        if society not in personalities and parent in personalities:
            personalities[society] = personalities[parent]


def branch(bridge: Model9Bridge, fork: int, actor, normalizer, critic, camp: int, society: int, alt: dict[str, Any],
           horizon: int, movement: bool, use_bootstrap: bool = False,
           personalities: dict[int, Any] | None = None) -> dict[str, Any]:
    """Roll one alternative out from the fork: its lineage's discounted return per objective (and the critic's
    per-objective bootstrap, added when ``use_bootstrap``), the whole world's discounted welfare (to tell a lineage
    gain that only takes from neighbours), and the heuristic's mix on a native branch."""
    step = bridge.restore(fork)
    personalities = None if personalities is None else dict(personalities)
    lineage = {society}
    returns = np.zeros(len(contracts.OBJECTIVES)); world = 0.0; discount = 1.0; native_mix = None
    for h in range(horizon):
        by_camp = {int(v["camp"]): v for v in step["villages"]}
        actions = override(policy_actions(actor, normalizer, step["villages"], movement, personality=personalities),
                           camp, alt, by_camp.get(camp), h == 0)
        step = bridge.step(actions)
        for v in step["villages"]:
            if v.get("parentSociety", 0) in lineage:
                lineage.add(int(v["society"]))
        inherit(personalities, step["villages"])
        returns += discount * component_rewards(step["villages"], lineage)
        world += discount * sum(float(v["welfare"]) for v in step["villages"])
        discount *= float(data.discount(np.asarray([step["years"]]))[0])
        if h == 0 and alt["kind"] == NATIVE:
            teacher = next((v.get("teacher") for v in step["villages"] if int(v["camp"]) == camp), None)
            if teacher is not None:
                s, f = teacher["strategic"][3], teacher["strategic"][4]
                native_mix = [(1 - f) * s, (1 - f) * (1 - s), f]
        if not any(not v["terminated"] for v in step["villages"]):
            break
    bootstraps = discount * lineage_values(critic, normalizer, step["villages"], lineage, personalities)
    total = returns + (np.nan_to_num(bootstraps) if use_bootstrap else 0.0)
    return {"returns": total, "return": float(returns[0]), "bootstrap": float(bootstraps[0]), "bootstraps": bootstraps,
            "world": world, "native_mix": native_mix}


def search_world(checkpoint: str, seed: int, withheld: int, warmup: int, config: SearchConfig) -> dict[str, Any]:
    """One world's main path with the policy, searching at sampled decision points."""
    torch.set_num_threads(1)
    actor, normalizer = load(Path(checkpoint)); critic = load_critic(Path(checkpoint))
    movement = bool(getattr(actor, "region_movement", False))
    rng = np.random.default_rng([seed, withheld, 9])
    draw = personality_sampler(config.personality, np.random.default_rng([seed, withheld, 11]))
    personalities: dict[int, Any] | None = None if config.personality == "default" else {}

    def assign(villages: list[dict[str, Any]]) -> None:  # first sight: inherit from a known parent, else draw
        if personalities is None:
            return
        inherit(personalities, villages)
        for v in villages:
            personalities.setdefault(int(v["society"]), draw().tolist())

    last = max(config.first_turn + 1, config.turns - config.horizon)
    turns = set(rng.choice(np.arange(config.first_turn, last), size=min(config.points, last - config.first_turn), replace=False).tolist())
    rows: list[dict[str, Any]] = []
    clock = time.perf_counter(); branches = 0
    with Model9Bridge() as bridge:
        step = bridge.reset(seed, withheld=withheld, movement=True)
        for _ in range(warmup):
            step = bridge.step({})
        for turn in range(config.turns):
            assign(step["villages"])
            if turn in turns:
                fork, decision = bridge.fork()
                base = policy_actions(actor, normalizer, decision["villages"], movement, personality=personalities)
                live = [v for v in decision["villages"] if v.get("observation") is not None and not v["terminated"]
                        and int(v["camp"]) in base]
                for v in [live[i] for i in rng.permutation(len(live))[:config.villages]]:
                    camp, society = int(v["camp"]), int(v["society"])
                    alts = alternatives(v, base[camp], config)
                    results = [branch(bridge, fork, actor, normalizer, critic, camp, society, a, config.horizon, movement,
                                      config.bootstrap, personalities) for a in alts]
                    branches += len(alts)
                    p = personalities.get(society) if personalities else [contracts.OBJECTIVE_WEIGHTS[k] for k in contracts.OBJECTIVES]
                    rows.append({"village": v, "alternatives": alts, "results": results, "turn": turn, "personality": p})
                step = bridge.restore(fork)
                bridge.drop(fork)
            step = bridge.step(policy_actions(actor, normalizer, step["villages"], movement, personality=personalities))
            if not any(not v["terminated"] for v in step["villages"]):
                break
    return {"seed": seed, "withheld": withheld, "rows": rows, "seconds": time.perf_counter() - clock, "branches": branches}


def label_arrays(worlds: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Label rows in the shard layout: observation arrays plus label_kind, label_slot, label_mix, the policy's and best
    scores, and every alternative's kind, slot and score (padded to MAX_ALTERNATIVES)."""
    villages, extra = [], {k: [] for k in ("seed", "turn", "camp", "label_kind", "label_slot", "label_mix", "score_policy",
                                           "score_best", "alt_kind", "alt_slot", "alt_score", "alt_return", "alt_bootstrap",
                                           "alt_world", "personality", "alt_returns", "alt_bootstraps")}
    width = len(contracts.OBJECTIVES)
    for world in worlds:
        for row in world["rows"]:
            p = np.asarray(row["personality"], dtype=np.float64)
            returns = np.stack([r["returns"] for r in row["results"]])
            scores = personality_score(returns, p)
            best = int(np.argmax(scores)); alt = row["alternatives"][best]
            mix = alt["mix"] if alt["kind"] == MIX else row["results"][best]["native_mix"] if alt["kind"] == NATIVE else None
            villages.append(row["village"])
            extra["seed"].append(world["seed"]); extra["turn"].append(row["turn"]); extra["camp"].append(int(row["village"]["camp"]))
            extra["label_kind"].append(alt["kind"]); extra["label_slot"].append(alt["slot"])
            extra["label_mix"].append(np.full(3, np.nan) if mix is None else np.asarray(mix, dtype=np.float64))
            extra["score_policy"].append(scores[0]); extra["score_best"].append(scores[best])
            pad = MAX_ALTERNATIVES - len(scores)
            extra["alt_kind"].append([a["kind"] for a in row["alternatives"]] + [-1] * pad)
            extra["alt_slot"].append([a["slot"] for a in row["alternatives"]] + [-1] * pad)
            extra["alt_score"].append(list(scores) + [np.nan] * pad)
            extra["alt_return"].append([r["return"] for r in row["results"]] + [np.nan] * pad)
            extra["alt_bootstrap"].append([r["bootstrap"] for r in row["results"]] + [np.nan] * pad)
            extra["alt_world"].append([r["world"] for r in row["results"]] + [np.nan] * pad)
            # The value vector of every alternative, so labels can be rescored for any personality later.
            extra["personality"].append(p.astype(np.float32))
            extra["alt_returns"].append(np.vstack([returns, np.full((pad, width), np.nan)]))
            extra["alt_bootstraps"].append(np.vstack([np.stack([r["bootstraps"] for r in row["results"]]),
                                                      np.full((pad, width), np.nan)]))
    if not villages:
        return {}
    arrays = observation_batch(villages)
    arrays.update({k: np.asarray(v) for k, v in extra.items()})
    return arrays


def summarize(labels: dict[str, np.ndarray], margin: float) -> dict[str, Any]:
    """How often and by how much search beats the policy, by winning kind."""
    if not labels:
        return {"rows": 0}
    gain = (labels["score_best"] - labels["score_policy"]) / np.maximum(1e-9, np.abs(labels["score_policy"]))
    kept = gain > margin
    out = {"rows": int(len(gain)), "beats_policy_share": float(np.mean(kept)), "margin": margin,
           "gain_median_when_kept": float(np.median(gain[kept])) if kept.any() else 0.0,
           "gain_p90": float(np.quantile(gain, 0.9))}
    for k, name in enumerate(KINDS):
        sel = kept & (labels["label_kind"] == k)
        out[f"wins_{name}"] = int(sel.sum())
    # Per alternative kind: mean relative score against the policy (paired).
    # Per alternative kind: mean relative score against the policy (paired), and the same for the horizon return alone
    # (no critic bootstrap), so a gain the critic invents shows as a gap between the two.
    def relative(key: str) -> np.ndarray:
        base = labels[key][:, :1]
        return (labels[key] - base) / np.maximum(1e-9, np.abs(base))
    rel, ret = relative("alt_score"), relative("alt_return")
    if "alt_world" in labels:
        # The winners' effect on the whole world: a lineage gain with a world loss takes from neighbours.
        best = np.nanargmax(np.where(labels["alt_kind"] >= 0, labels["alt_score"], -np.inf), axis=1)
        world = labels["alt_world"]
        world_gain = (world[np.arange(len(best)), best] - world[:, 0]) / np.maximum(1e-9, np.abs(world[:, 0]))
        out["winner_world_gain_median"] = float(np.median(world_gain[kept])) if kept.any() else 0.0
        out["winner_world_loss_share"] = float(np.mean(world_gain[kept] < 0)) if kept.any() else 0.0
    for k, name in enumerate(KINDS[1:], start=1):
        sel = labels["alt_kind"] == k
        out[f"mean_gain_{name}"] = float(np.nanmean(rel[sel])) if sel.any() else None
        out[f"mean_return_gain_{name}"] = float(np.nanmean(ret[sel])) if sel.any() else None
        out[f"count_{name}"] = int(sel.sum())
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seeds", default="1000-1011", help="TRAIN seed range (never selection or gate seeds)")
    parser.add_argument("--valley-seeds", type=Path, default=None, help="valley fixture bank (lines 'seed,warmup')")
    parser.add_argument("--withheld", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--margin", type=float, default=0.005, help="relative gain a label must clear (summary only)")
    parser.add_argument("--out", type=Path, required=True)
    for name, value in asdict(SearchConfig()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value) if not isinstance(value, bool) else int, default=value)
    args = parser.parse_args()
    config = SearchConfig(**{k: (bool(getattr(args, k)) if isinstance(v, bool) else getattr(args, k)) for k, v in asdict(SearchConfig()).items()})
    a, b = (int(x) for x in args.seeds.split("-"))
    worlds = [(s, args.withheld, 0) for s in range(a, b + 1)]
    if args.valley_seeds:
        worlds += [(int(s), 0, int(w)) for s, w in (line.split(",") for line in args.valley_seeds.read_text().split() if line.strip())]
    args.out.mkdir(parents=True, exist_ok=True)
    clock = time.time()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(worlds))) as pool:
        results = list(pool.map(search_world, [args.checkpoint] * len(worlds), [w[0] for w in worlds], [w[1] for w in worlds],
                                [w[2] for w in worlds], [config] * len(worlds)))
    labels = label_arrays(results)
    if labels:
        np.savez_compressed(args.out / "labels.npz", **labels)
    summary = {"checkpoint": args.checkpoint, "config": asdict(config), "worlds": len(worlds), "seconds": time.time() - clock,
               "branches": sum(r["branches"] for r in results),
               "seconds_per_branch": sum(r["seconds"] for r in results) / max(1, sum(r["branches"] for r in results)),
               **summarize(labels, args.margin)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
