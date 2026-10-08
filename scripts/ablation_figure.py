"""Ablation examples figure from the npz written by ablation.py. usage: ablation_figure.py ABLATION.npz OUT.png"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

d = np.load(sys.argv[1])
c, les, suc, names = d["c"], d["lesions"], d["success"], [str(n) for n in d["strategies"]]
label = {"surrogate": "surrogate's choice", "border_zone": "border-zone heuristic", "random": "random line"}
colour = {"surrogate": [0.85, 0.15, 0.15], "border_zone": [0.95, 0.6, 0.2], "random": [0.4, 0.4, 0.4]}
order = np.argsort(~suc[:, 0])[:4]
fig, axes = plt.subplots(len(order), 3, figsize=(7.2, 2.5 * len(order)))
for r, i in enumerate(order):
    for k, name in enumerate(names):
        a = axes[r, k]
        rgb = np.stack([c[i]] * 3, -1).astype(float)
        rgb[les[i, k]] = colour[name]
        a.imshow(rgb, origin="lower")
        a.set_xticks([])
        a.set_yticks([])
        ok = bool(suc[i, k])
        a.set_title(f"{label[name]}: {'stops re-entry' if ok else 'still inducible'}", fontsize=7.5,
                    color="#1e8449" if ok else "#922b21")
fig.suptitle("One ablation line per tissue, outcome checked by re-simulation", fontsize=9)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=140)
