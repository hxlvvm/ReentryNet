"""Verification round E1/E1b/E2/E5 for the ablation claim. usage: verify_ablation.py DATA OUT.npz GROUP CKPT [CKPT ...]

GROUP single: surrogate, surrogate excluding lesions within 20 px of S2, closest-to-S2, isthmus heuristic, random,
              plus S1-only side effects. GROUP top5 / random5: five simulated lines per tissue, success if any works.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ablation import INTERVALS, N_TISSUES, candidates, surrogate_scores  # noqa: E402
from reentrynet.net import ReentryUNet  # noqa: E402
from reentrynet.sim import run_protocol, s1_activation  # noqa: E402

S2_CENTRE, NEAR = (64, 3), 20.0
d, out, group, ckpts = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
dev = "cuda"
parts = [np.load(Path(d) / f"block_{s}.npz") for s in (1008, 1009)]
C = np.concatenate([p["c"] for p in parts]).astype(np.float32)
ind = np.concatenate([p["inducible"] for p in parts])
rng = np.random.default_rng(7)
pick = rng.choice(np.where(ind)[0], N_TISSUES, replace=False)
C = C[pick]
masks, meta = candidates()
yy, xx = np.mgrid[:128, :128]
dist_s2 = np.array([np.hypot(yy[m] - S2_CENTRE[0], xx[m] - S2_CENTRE[1]).min() for m in masks])
models = []
for k in ckpts:
    m = ReentryUNet().to(dev).eval()
    m.load_state_dict(torch.load(k, map_location=dev)["state"])
    models.append(m)
grow = np.array([ndimage.binary_dilation(m) for m in masks])
rng2 = np.random.default_rng(11 if group == "single" else 12 if group == "top5" else 13)
picks = {}
for i, c in enumerate(C):
    valid = np.array([(c[m] > 0).any() for m in masks])
    scores = surrogate_scores(models, c, masks, dev)
    scores[~valid] = np.inf
    order = np.argsort(scores)
    if group == "single":
        far = scores.copy()
        far[dist_s2 < NEAR] = np.inf
        lab, n = ndimage.label(c == 0)
        sizes = np.bincount(lab.ravel())
        big = np.flatnonzero(sizes >= 10)
        big = big[big > 0]
        touch = np.array([len(np.intersect1d(np.unique(lab[g]), big)) for g in grow]).astype(float)
        touch[~valid] = -1
        cand = np.flatnonzero(touch == touch.max())
        bz = np.array([((c[masks[k]] > 0) & (c[masks[k]] < 1)).sum() for k in cand])
        near_order = np.where(valid, dist_s2, np.inf)
        choice = {"surrogate": int(order[0]), "surrogate_far": int(np.argmin(far)), "closest_to_s2": int(np.argmin(near_order)),
                  "isthmus": int(cand[np.argmax(bz)]), "random": int(rng2.choice(np.flatnonzero(valid)))}
    elif group == "top5":
        choice = {f"top{j}": int(order[j]) for j in range(5)}
    else:
        r5 = rng2.choice(np.flatnonzero(valid), 5, replace=False)
        choice = {f"rand{j}": int(r5[j]) for j in range(5)}
    for k, v in choice.items():
        picks.setdefault(k, []).append(v)
print("search done", flush=True)


def simulate(tissues):
    return np.concatenate([run_protocol(torch.from_numpy(tissues[s:s + 200]).to(dev), INTERVALS)["inducible"].cpu().numpy()
                           for s in range(0, len(tissues), 200)])


def ablate(ch):
    T = C.copy()
    for i, k in enumerate(ch):
        T[i][masks[k]] = 0.0
    return T


res = {"picks": {k: np.array(v) for k, v in picks.items()}, "dist_s2": dist_s2}
for k, ch in picks.items():
    res[f"success_{k}"] = ~simulate(ablate(ch))
    print(k, int(res[f"success_{k}"].sum()), "/", len(C), flush=True)
if group == "single":
    def s1(tissues):
        return np.concatenate([s1_activation(torch.from_numpy(tissues[s:s + 200]).to(dev)).cpu().numpy() for s in range(0, len(tissues), 200)])
    base = s1(C)
    res["s1_base_activated"] = np.isfinite(base).sum((1, 2)) / (C > 0).sum((1, 2))
    res["s1_base_p95"] = np.nanpercentile(base.reshape(len(C), -1), 95, axis=1)
    for k in ("surrogate", "surrogate_far", "random"):
        T = ablate(picks[k])
        a = s1(T)
        res[f"s1_{k}_activated"] = np.isfinite(a).sum((1, 2)) / (T > 0).sum((1, 2))
        res[f"s1_{k}_p95"] = np.nanpercentile(a.reshape(len(C), -1), 95, axis=1)
np.savez_compressed(out, **res)
print("saved", out)
