"""Verification round E3/E6 for the labels. usage: verify_labels.py DATA OUT.json TASK(e3|dt|dx)"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reentrynet.sim import run_protocol  # noqa: E402

INTERVALS = [41.0, 43.0, 45.0, 48.0, 52.0, 60.0]
d, out, task = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
dev = "cuda"


def ind(c, batch=200, **kw):
    return np.concatenate([run_protocol(torch.from_numpy(c[s:s + batch]).to(dev), INTERVALS, **kw)["inducible"].cpu().numpy()
                           for s in range(0, len(c), batch)])


res = {"task": task}
if task == "e3":
    parts = [np.load(d / f"block_{s}.npz") for s in (1008, 1009)]
    C = np.concatenate([p["c"] for p in parts]).astype(np.float32)
    lab = np.concatenate([p["inducible"] for p in parts])
    pick = np.random.default_rng(7).choice(np.where(lab)[0], 400, replace=False)
    healed = np.where(C[pick] < 1, 1.0, C[pick]).astype(np.float32)
    res["healed_inducible"] = float(ind(healed).mean())
    sub = C[pick[:64]]
    res["n_long"] = 64
    res["still_active_450"] = float(ind(sub).mean())
    res["still_active_1200"] = float(ind(sub, tail=1200.0).mean())
else:
    p = np.load(d / "block_1007.npz")
    n = 512 if task == "dt" else 256
    C, lab = p["c"][:n].astype(np.float32), p["inducible"][:n]
    if task == "dt":
        new = ind(C, dt=0.025)
    else:
        fine = np.kron(C, np.ones((1, 2, 2), np.float32))
        new = ind(fine, batch=32, dt=0.025, dx=0.5)
    res.update({"n": int(n), "flips": int((new != lab).sum()), "flip_rate": float((new != lab).mean()),
                "inducible_original": float(lab.mean()), "inducible_new": float(new.mean())})
print(json.dumps(res, indent=1))
json.dump(res, open(out, "w"), indent=1)
