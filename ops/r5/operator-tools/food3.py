import glob, os, numpy as np, torch
from anthropocene_rl.autonomous_v7 import contracts
from anthropocene_rl.autonomous_v7.labor_privileged import LABOR_PRIVILEGED_LAYOUT_V1 as L
ds = "/home/suyashbhatt11_gmail_com/r5/fit8-b01"
names = [n for n, _, _ in contracts.SETS]; _, R, Wd = contracts.SETS[names.index("trade")]
food = contracts.TRADE_GROUPS.index("food")
nz = torch.load("/home/suyashbhatt11_gmail_com/r5/actor-normalizer.pt", weights_only=False)["normalizer"]
mean, std = np.array(nz["trade_mean"]), np.array(nz["trade_std"])
ec = L.family("economy").features; mi = ec.index("meals-per-turn")
out = []
for w in sorted(glob.glob(os.path.join(ds, "world-*"))):
    t = np.load(os.path.join(w, "trade.npy")).reshape(-1, R, Wd).astype(np.float64)
    y = t * std + mean; x = np.sign(y) * np.expm1(np.abs(y))
    p = np.load(os.path.join(w, 'trade_present.npy')).reshape(len(t), R).astype(bool)
    imp = x[:, food, 6]; idx = np.nonzero((np.abs(imp) > 1e-6) & p[:, food] & (np.abs(t[:, food, 6]) > 1e-9))[0]
    eco = None
    for cand in ("privileged_economy.npy",):
        p = os.path.join(w, cand)
        if os.path.exists(p): eco = np.load(p).reshape(len(t), -1)
    for i in idx:
        out.append((os.path.basename(w), int(i), float(imp[i]), None if eco is None else float(eco[i, mi])))
print("nonzero food-imported rows:", len(out), "worlds:", sorted({o[0] for o in out}))
for o in out[:12]: print("  ", o)
v = np.array([o[2] for o in out]); print("imported raw: min %.3g median %.3g max %.3g" % (v.min(), np.median(v), v.max()) if len(v) else "none")
pass #, [f for f in os.listdir(sorted(glob.glob(os.path.join(ds,'world-*')))[0]) if f.startswith('privileged')][:12])
