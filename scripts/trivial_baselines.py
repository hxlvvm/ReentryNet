"""Trivial baselines for the two map outputs (added after the frozen evaluation; reported as such)."""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
d = Path(sys.argv[1])
tr = [np.load(d / f"block_{s}.npz") for s in range(1000, 1007)]
act_tr = np.concatenate([p["s1_act"] for p in tr]).astype(np.float32)
mean_map = np.nanmean(act_tr, 0)
res = {}
for name, files in (("test", ["block_1008", "block_1009"]), ("rings", ["rings_2000"])):
    parts = [np.load(d / f"{f}.npz") for f in files]
    c = np.concatenate([p["c"] for p in parts]).astype(np.float32)
    rm = np.concatenate([p["reentry_map"] for p in parts]).astype(np.float32)
    ind = np.concatenate([p["inducible"] for p in parts])
    a = np.concatenate([p["s1_act"] for p in parts]).astype(np.float32)
    idx = np.where(ind)[0]
    true = rm[idx] > 0.2
    alive = c[idx] > 0
    inter = (true & alive).sum((1, 2))
    dice_all = float(np.mean(2 * inter / (true.sum((1, 2)) + alive.sum((1, 2)))))
    valid = np.isfinite(a) & np.isfinite(mean_map)[None]
    res[name] = {"dice_all_living_tissue": dice_all,
                 "reentry_map_coverage_median": float(np.median(true.sum((1, 2)) / alive.sum((1, 2)))),
                 "activation_mae_pixel_mean_baseline": float(np.abs(a - mean_map[None])[valid].mean()),
                 "activation_valid_fraction_median": float(np.median(np.isfinite(a).mean((1, 2))))}
print(json.dumps(res, indent=1))
json.dump(res, open(sys.argv[2], "w"), indent=1)
