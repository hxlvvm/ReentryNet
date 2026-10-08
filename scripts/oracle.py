"""Upper bound: simulate every candidate on a thinned grid for a slice of the ablation tissues.

usage: oracle.py DATA OUT.json START COUNT CKPT [CKPT ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ablation import INTERVALS, N_TISSUES, candidates, surrogate_scores  # noqa: E402
from reentrynet.net import ReentryUNet  # noqa: E402
from reentrynet.sim import run_protocol  # noqa: E402

d, out, start, count, ckpts = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5:]
dev = "cuda"
parts = [np.load(Path(d) / f"block_{s}.npz") for s in (1008, 1009)]
C = np.concatenate([p["c"] for p in parts]).astype(np.float32)
ind = np.concatenate([p["inducible"] for p in parts])
rng = np.random.default_rng(7)
pick = rng.choice(np.where(ind)[0], N_TISSUES, replace=False)
C = C[pick][start:start + count]
masks, _ = candidates(step=12)
models = []
for k in ckpts:
    m = ReentryUNet().to(dev).eval()
    m.load_state_dict(torch.load(k, map_location=dev)["state"])
    models.append(m)
rows = []
for i, c in enumerate(C):
    valid = np.flatnonzero([(c[m] > 0).any() for m in masks])
    T = np.repeat(c[None], len(valid), 0)
    for j, k in enumerate(valid):
        T[j][masks[k]] = 0.0
    works = np.concatenate([run_protocol(torch.from_numpy(T[s:s + 200]).to(dev), INTERVALS)["inducible"].cpu().numpy()
                            for s in range(0, len(T), 200)]) == 0
    scores = surrogate_scores(models, c, masks[valid], dev)
    rows.append({"tissue": start + i, "candidates": int(len(valid)), "any_works": bool(works.any()),
                 "fraction_working": float(works.mean()), "surrogate_choice_works": bool(works[np.argmin(scores)]),
                 "surrogate_top5_any_works": bool(works[np.argsort(scores)[:5]].any())})
    print(rows[-1], flush=True)
json.dump(rows, open(out, "w"), indent=1)
