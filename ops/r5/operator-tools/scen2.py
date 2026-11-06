import glob
import json
import os
import sys

import numpy as np
import torch

from anthropocene_rl.autonomous_v7 import contracts

ds = sys.argv[1]
names = [n for n, _, _ in contracts.SETS]
si = names.index("trade")
_, R, Wd = contracts.SETS[si]
food = contracts.TRADE_GROUPS.index("food")
nz = torch.load("/home/suyashbhatt11_gmail_com/r5/actor-normalizer.pt", weights_only=False)[
    "normalizer"
]
mean, std = np.array(nz["trade_mean"]), np.array(nz["trade_std"])
tot = 0
pres_rows = 0
nonzero = {c: 0 for c in range(Wd)}
anygood = np.zeros((R, Wd), int)
counters = {}
for w in sorted(glob.glob(os.path.join(ds, "world-*"))):
    t = np.load(os.path.join(w, "trade.npy")).reshape(-1, R, Wd).astype(np.float64)
    p = np.load(os.path.join(w, "trade_present.npy")).reshape(len(t), R).astype(bool)
    y = t * std + mean
    x = np.sign(y) * np.expm1(np.abs(y))
    tot += len(t)
    pres_rows += int(p[:, food].sum())
    for c in range(Wd):
        nonzero[c] += int(((np.abs(x[:, food, c]) > 1e-6) & p[:, food]).sum())
    anygood += ((np.abs(x) > 1e-6) & p[:, :, None]).sum(0)
    m = json.load(open(os.path.join(w, "meta.json")))

    def walk(o, pre=""):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{pre}.{k}" if pre else k)
        elif isinstance(o, (int, float)) and "trade" in pre.lower():
            counters[pre] = counters.get(pre, 0) + o

    walk(m.get("counters", {}))
print(f"rows {tot}; food row present {pres_rows}")
print("food row: rows with nonzero raw value per column:", nonzero)
print("rows with nonzero raw value, per trade group x column (groups", contracts.TRADE_GROUPS, "):")
for g, row in zip(contracts.TRADE_GROUPS, anygood, strict=False):
    print("  ", g, row.tolist())
print(
    "trade-related meta counters summed over worlds:", json.dumps(counters, sort_keys=True)[:2000]
)
