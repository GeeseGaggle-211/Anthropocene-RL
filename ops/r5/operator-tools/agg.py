import json
import os
import re

R = os.path.expanduser("~/g0")


def J(n):
    return json.load(open(f"{R}/{n}"))


def small(d, lim=20000):
    if not isinstance(d, dict):
        return d
    return {k: v for k, v in d.items() if len(json.dumps(v)) < lim}


def L(n):
    return [l.rstrip("\n") for l in open(f"{R}/{n}") if not l.startswith("{")]


out = {
    "source": "anthropocene-trainer-60:~/g0",
    "generated_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
}
out["actor_fidelity"] = {
    "log": L("eval-final-actor.log"),
    "json_summary": small(J("eval-final-actor.json")),
}
crit = []
for l in L("eval-final-critic.log"):
    m = re.match(
        r"\s+(\w+)\s+(\w+)\s+(PASS|FAIL)\s+nMAE=(\S+) bias=(\S+) rank=(\S+) pairs=(\d+) groups=(\d+) worlds=(\d+)",
        l,
    )
    if m:
        h, s, g, n, b, r, p, gr, w = m.groups()
        crit.append(
            {
                "head": h,
                "slice": s,
                "gate": g,
                "nMAE": float(n),
                "bias": float(b),
                "rank": None if r == "n/a" else float(r),
                "pairs": int(p),
                "groups": int(gr),
                "worlds": int(w),
            }
        )
out["critic_calibration"] = {
    "rows": crit,
    "log": L("eval-final-critic.log"),
    "json_summary": small(J("eval-final-critic.json")),
}
out["autonomous_eval"] = {
    "log": L("eval-final-auto.log"),
    "json_summary": small(J("eval-final-auto.json")),
}
out["critic3_calibration_log"] = (
    L("eval-critic3.log")[-30:] if os.path.exists(f"{R}/eval-critic3.log") else None
)
g4 = []
for l in open(f"{R}/dagger/driver.log"):
    if (
        re.search(r"round \d+ done depth", l)
        or ("round " in l and " collected " in l)
        or "g4 complete" in l
    ):
        g4.append(l.rstrip("\n"))
out["g4_rounds"] = g4
out["pipeline_log"] = [l.rstrip("\n") for l in open(f"{R}/pipeline.log")][-12:]
print(json.dumps(out, indent=1, allow_nan=True))
