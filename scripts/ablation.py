"""Surrogate-guided ablation vs simple strategies, verified by re-simulation. usage: ablation.py DATA OUT.json CKPT [CKPT ...]"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reentrynet.net import ReentryUNet  # noqa: E402
from reentrynet.sim import run_protocol  # noqa: E402

INTERVALS = [41.0, 43.0, 45.0, 48.0, 52.0, 60.0]
N_TISSUES, LENGTH, WIDTH = 400, 20, 3


def candidates(shape=(128, 128), step=6):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    masks, meta = [], []
    for cy in range(6, shape[0] - 5, step):
        for cx in range(14, shape[1] - 5, step):
            for ang in (0, 45, 90, 135):
                t = np.deg2rad(ang)
                m = np.zeros(shape, bool)
                for s in np.linspace(-LENGTH / 2, LENGTH / 2, 4 * LENGTH):
                    m |= (yy - (cy + s * np.sin(t))) ** 2 + (xx - (cx + s * np.cos(t))) ** 2 <= (WIDTH / 2) ** 2
                m[:, :10] = False
                masks.append(m)
                meta.append((cy, cx, ang))
    return np.stack(masks), meta


def wilson(x, n, z=1.96):
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


@torch.no_grad()
def surrogate_scores(models, c, masks, dev):
    out = []
    mt = torch.from_numpy(masks).to(dev)
    base = torch.from_numpy(c).to(dev)
    for s in range(0, len(masks), 256):
        batch = base.expand(len(mt[s:s + 256]), -1, -1).clone()
        batch[mt[s:s + 256]] = 0.0
        out.append(torch.stack([torch.sigmoid(m(batch)[0]) for m in models]).mean(0).cpu().numpy())
    return np.concatenate(out)


def main():
    d, out_json, ckpts = sys.argv[1], sys.argv[2], sys.argv[3:]
    dev = "cuda"
    parts = [np.load(Path(d) / f"block_{s}.npz") for s in (1008, 1009)]
    C = np.concatenate([p["c"] for p in parts]).astype(np.float32)
    ind = np.concatenate([p["inducible"] for p in parts])
    rng = np.random.default_rng(7)
    pick = rng.choice(np.where(ind)[0], N_TISSUES, replace=False)
    C = C[pick]
    masks, meta = candidates()
    models = []
    for k in ckpts:
        m = ReentryUNet().to(dev).eval()
        m.load_state_dict(torch.load(k, map_location=dev)["state"])
        models.append(m)
    t0 = time.time()
    choice = {"surrogate": [], "border_zone": [], "random": []}
    pred = []
    for i, c in enumerate(C):
        valid = np.array([(c[m] > 0).any() for m in masks])
        scores = surrogate_scores(models, c, masks, dev)
        scores[~valid] = np.inf
        bz = np.array([((c[m] > 0) & (c[m] < 1)).sum() for m in masks]).astype(float)
        bz[~valid] = -1
        choice["surrogate"].append(int(np.argmin(scores)))
        choice["border_zone"].append(int(rng.choice(np.flatnonzero(bz == bz.max()))))
        choice["random"].append(int(rng.choice(np.flatnonzero(valid))))
        pred.append(float(scores[choice["surrogate"][-1]]))
    t_search = (time.time() - t0) / len(C)
    print(f"surrogate search: {t_search:.2f} s per tissue for {len(masks)} candidates", flush=True)

    def simulate(tissues):
        res = []
        for s in range(0, len(tissues), 200):
            res.append(run_protocol(torch.from_numpy(tissues[s:s + 200]).to(dev), INTERVALS))
        return np.concatenate([r["inducible"].cpu().numpy() for r in res]), np.concatenate([r["s1_act"].cpu().numpy() for r in res])

    base_ind, base_act = simulate(C)
    alive = C > 0
    result = {"n": len(C), "candidates": len(masks), "lesion": {"length": LENGTH, "width": WIDTH},
              "sanity_unablated_still_inducible": float(base_ind.mean()), "surrogate_search_seconds_per_tissue": t_search,
              "strategies": {}}
    success = {}
    for name, ch in choice.items():
        T = C.copy()
        for i, k in enumerate(ch):
            T[i][masks[k]] = 0.0
        ind_after, act_after = simulate(T)
        success[name] = ~ind_after
        unreached = [float((np.isnan(act_after[i]) & (T[i] > 0)).sum() / max((T[i] > 0).sum(), 1)
                           - (np.isnan(base_act[i]) & alive[i]).sum() / max(alive[i].sum(), 1)) for i in range(len(C))]
        x = int(success[name].sum())
        result["strategies"][name] = {"made_non_inducible": x, "rate": x / len(C), "ci": wilson(x, len(C)),
                                      "median_extra_unreached_fraction": float(np.median(unreached))}
        print(name, x, "/", len(C), flush=True)
    for a, b in (("surrogate", "border_zone"), ("surrogate", "random")):
        n01 = int((success[a] & ~success[b]).sum())
        n10 = int((~success[a] & success[b]).sum())
        p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        result[f"mcnemar_{a}_vs_{b}"] = {"only_first_succeeds": n01, "only_second_succeeds": n10, "p": float(p),
                                         "difference_points": 100 * (success[a].mean() - success[b].mean())}
    sp = np.array(pred)
    result["surrogate_predicted_vs_actual"] = {"mean_predicted_inducibility": float(sp.mean()),
                                               "actual_still_inducible": float(1 - success["surrogate"].mean())}
    ex = rng.choice(len(C), 6, replace=False)
    np.savez_compressed(Path(out_json).with_suffix(".npz"), c=C[ex], lesions=np.stack([[masks[choice[n][i]] for n in choice] for i in ex]),
                        success=np.stack([[success[n][i] for n in choice] for i in ex]), strategies=np.array(list(choice)))
    json.dump(result, open(out_json, "w"), indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
